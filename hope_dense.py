"""Hope neural learning module (Behrouz et al., Nested Learning, NeurIPS 2025, arXiv 2512.24695, Sec. 7-8) in plain PyTorch,
chunk-parallel, for short token sequences (here: windows of heartbeats).

Two-layer residual memory used everywhere (Eq. 89):  M(z) = z + W1 gelu(W2 z)
Inner objective (L2 regression):  l = 1/2 || M(k) - target ||^2, gradients in closed form:
    a = W2 k, h = gelu(a), e = M(k) - target,  dW1 = e h^T,  delta = (W1^T e) * gelu'(a),  dW2 = delta k^T
Chunk-wise update (Sec. 8.2): all keys/values/rates of a chunk are generated with the memory state at the end of the previous
chunk, the gradient is taken at that state, averaged over the chunk (scaled by the per-token rate eta_t), bounded to Frobenius
norm <= 1, and applied with the retention gate:  W <- (1 - alpha_c) W - G.  Tokens of a chunk read the updated state.

SelfModifyingTitans (Eqs. 83-85, 90):  k, v, eta, alpha are produced by their own memories M_k, M_v, M_eta, M_alpha; every
memory (including the output memory M_mem) regresses M(k_t) onto its self-generated value v_hat = M(v_t); q is static.
ContinuumMemory (Eqs. 70-71, sequential variant): a chain of MLP blocks, level l updated in context every C_l tokens; the
in-context objective of every level is next-token (next-beat) prediction, the self-supervised signal available at test time.
Variants that revert to older mechanisms (ablations):
    mixer='titans'  static k, v, eta, alpha projections, deep memory regressing M(k) onto v (Titans without momentum)
    mixer='delta'   static projections, linear matrix memory, delta rule  W <- (1 - alpha) W - eta (W k - v) k^T (Gated DeltaNet)
    mixer='none'    no sequence memory (local convolution only)
    cms=False       static feed-forward block instead of the continuum memory"""
import math, torch, torch.nn as nn, torch.nn.functional as F


def gelu_grad(a):
    cdf = 0.5 * (1.0 + torch.erf(a / math.sqrt(2.0))); pdf = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    return cdf + a * pdf


def mlp_mem(z, W1, W2):
    """z (..., b, d), W1 (..., d, m), W2 (..., m, d) -> z + gelu(z W2^T) W1^T"""
    return z + F.gelu(z @ W2.transpose(-1, -2)) @ W1.transpose(-1, -2)


def mlp_grad(k, target, W1, W2, rate):
    """Chunk gradient of 1/2 sum_t rate_t ||M(k_t) - target_t||^2 / b with respect to (W1, W2); k, target (..., b, d), rate (..., b, 1)."""
    a = k @ W2.transpose(-1, -2); h = F.gelu(a)
    e = k + h @ W1.transpose(-1, -2) - target
    delta = (e @ W1) * gelu_grad(a)
    b = k.shape[-2]
    return (rate * e).transpose(-1, -2) @ h / b, (rate * delta).transpose(-1, -2) @ k / b, e


def bounded(*G):
    n = torch.sqrt(sum((g * g).sum((-1, -2)) for g in G) + 1e-12)
    s = 1.0 / torch.clamp(n, min=1.0)
    return [g * s[..., None, None] for g in G]


class SelfModifyingTitans(nn.Module):
    """Multi-head self-referential memory over (B, L, D). Returns (B, L, D) and the per-token surprise (B, L)."""
    NAMES = ['k', 'v', 'eta', 'alpha', 'mem']

    def __init__(self, dim, heads=4, d_head=32, hidden=64, chunk=2, mixer='hope', eta_max=1.0, alpha_max=0.5):
        super().__init__()
        self.H, self.d, self.m, self.C, self.mixer = heads, d_head, hidden, chunk, mixer
        self.eta_max, self.alpha_max = eta_max, alpha_max
        self.inp = nn.Linear(dim, heads * d_head, bias=False)          # static input projection to the heads
        self.q = nn.Linear(dim, heads * d_head, bias=False)            # q is the only non-adaptive projection (Eq. 85)
        if mixer == 'hope':
            # meta-learned initial states of the five memories (outer level)
            # the five memories are stacked on one axis (order NAMES) and updated in one batched step
            self.W1 = nn.Parameter(torch.zeros(len(self.NAMES), heads, d_head, hidden))
            self.W2 = nn.Parameter(torch.randn(len(self.NAMES), heads, hidden, d_head) / math.sqrt(d_head))
            self.r_eta = nn.Parameter(torch.zeros(heads, d_head)); self.b_eta = nn.Parameter(torch.full((heads,), -1.0))
            self.r_alpha = nn.Parameter(torch.zeros(heads, d_head)); self.b_alpha = nn.Parameter(torch.full((heads,), -3.0))
        else:
            self.kv = nn.Linear(dim, 2 * heads * d_head, bias=False)
            self.gates = nn.Linear(dim, 2 * heads); nn.init.zeros_(self.gates.weight)
            with torch.no_grad(): self.gates.bias[:heads] = -1.0; self.gates.bias[heads:] = -3.0
            if mixer == 'titans':
                self.W1m = nn.Parameter(torch.zeros(heads, d_head, hidden)); self.W2m = nn.Parameter(torch.randn(heads, hidden, d_head) / math.sqrt(d_head))
            elif mixer == 'delta':
                self.Wl = nn.Parameter(torch.zeros(heads, d_head, d_head))
        self.out = nn.Linear(heads * d_head, dim, bias=False)
        self.norm = nn.LayerNorm(heads * d_head)

    def heads(self, t, B, L):
        return t.view(B, L, self.H, self.d).transpose(1, 2)               # (B, H, L, d)

    def forward(self, x):
        B, L, _ = x.shape; H = self.H
        q = F.normalize(self.heads(self.q(x), B, L), dim=-1)
        n = math.ceil(L / self.C); ys, surp = [], []
        if self.mixer == 'hope':
            xh = self.heads(self.inp(x), B, L)
            W1 = self.W1.unsqueeze(0).expand(B, -1, -1, -1, -1); W2 = self.W2.unsqueeze(0).expand(B, -1, -1, -1, -1)   # (B, 5, H, ., .)
            for c in range(n):
                sl = slice(c * self.C, min(L, (c + 1) * self.C)); xc = xh[:, :, sl]
                G = mlp_mem(xc.unsqueeze(1), W1, W2)                                         # Eq. 83 with the previous-chunk states
                k = F.normalize(G[:, 0], dim=-1); v = G[:, 1]
                eta = self.eta_max * torch.sigmoid((G[:, 2] * self.r_eta[None, :, None]).sum(-1) + self.b_eta[None, :, None])      # (B, H, b)
                alpha = self.alpha_max * torch.sigmoid((G[:, 3] * self.r_alpha[None, :, None]).sum(-1)
                                                       + self.b_alpha[None, :, None]).mean(-1)                                    # (B, H)
                vhat = mlp_mem(v.unsqueeze(1), W1, W2)                                       # Eq. 84: self-generated values, all memories
                g1, g2, e = mlp_grad(k.unsqueeze(1), vhat, W1, W2, eta[:, None, :, :, None])
                g1, g2 = bounded(g1, g2)
                a = alpha[:, None, :, None, None]
                W1 = (1 - a) * W1 - g1; W2 = (1 - a) * W2 - g2
                surp.append(e[:, 4].norm(dim=-1))
                ys.append(mlp_mem(q[:, :, sl], W1[:, 4], W2[:, 4]))                          # read the updated output memory
        elif self.mixer in ('titans', 'delta'):
            k, v = self.kv(x).view(B, L, 2, H, self.d).permute(2, 0, 3, 1, 4); k = F.normalize(k, dim=-1)
            g = self.gates(x).view(B, L, 2, H).permute(2, 0, 3, 1)
            eta_all = self.eta_max * torch.sigmoid(g[0]); alpha_all = self.alpha_max * torch.sigmoid(g[1])
            if self.mixer == 'titans':
                W1, W2 = self.W1m.unsqueeze(0).expand(B, -1, -1, -1), self.W2m.unsqueeze(0).expand(B, -1, -1, -1)
            else:
                W = self.Wl.unsqueeze(0).expand(B, -1, -1, -1)
            for c in range(n):
                sl = slice(c * self.C, min(L, (c + 1) * self.C)); kc, vc = k[:, :, sl], v[:, :, sl]
                eta = eta_all[:, :, sl, None]; a = alpha_all[:, :, sl].mean(-1)[..., None, None]
                if self.mixer == 'titans':
                    g1, g2, e = mlp_grad(kc, vc, W1, W2, eta); g1, g2 = bounded(g1, g2)
                    W1 = (1 - a) * W1 - g1; W2 = (1 - a) * W2 - g2
                    ys.append(mlp_mem(q[:, :, sl], W1, W2))
                else:
                    e = kc @ W.transpose(-1, -2) - vc                                        # delta rule
                    (G,) = bounded((eta * e).transpose(-1, -2) @ kc / kc.shape[-2])
                    W = (1 - a) * W - G
                    ys.append(q[:, :, sl] @ W.transpose(-1, -2))
                surp.append(e.norm(dim=-1))
        else:
            return torch.zeros_like(x), torch.zeros(B, L, device=x.device)
        y = torch.cat(ys, 2).transpose(1, 2).reshape(B, L, H * self.d)
        return self.out(self.norm(y)), torch.cat(surp, 2).mean(1)


class ContinuumMemory(nn.Module):
    """Sequential CMS: levels with chunk sizes `chunks`, each M_l(z) = z + W1 gelu(W2 LN(z)), updated in context every C_l
    tokens by next-token prediction  1/2 ||M_l(z_t) - z_{t+1}||^2  (the last token has no target).  cms=False -> static FFNs."""

    def __init__(self, dim, hidden=256, chunks=(8, 4, 2), adaptive=True, eta_max=1.0):
        super().__init__()
        self.chunks, self.adaptive, self.eta_max = chunks, adaptive, eta_max
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in chunks])
        self.W1 = nn.ParameterList([nn.Parameter(torch.zeros(dim, hidden)) for _ in chunks])
        self.W2 = nn.ParameterList([nn.Parameter(torch.randn(hidden, dim) / math.sqrt(dim)) for _ in chunks])
        self.rate = nn.ModuleList([nn.Linear(dim, 1) for _ in chunks])
        for r in self.rate: nn.init.zeros_(r.weight); nn.init.constant_(r.bias, -1.0)

    def forward(self, z):
        B, L, D = z.shape
        for l, C in enumerate(self.chunks):
            W1 = self.W1[l].unsqueeze(0).expand(B, -1, -1); W2 = self.W2[l].unsqueeze(0).expand(B, -1, -1)
            if not self.adaptive:
                z = z + F.gelu(self.ln[l](z) @ W2.transpose(-1, -2)) @ W1.transpose(-1, -2); continue
            u = self.ln[l](z)
            tgt = torch.cat([z[:, 1:], z[:, -1:]], 1); valid = torch.ones(B, L, 1, device=z.device); valid[:, -1] = 0
            eta = self.eta_max * torch.sigmoid(self.rate[l](u)) * valid
            outs = []
            for c in range(math.ceil(L / C)):
                sl = slice(c * C, min(L, (c + 1) * C)); uc = u[:, sl]
                a = uc @ W2.transpose(-1, -2); h = F.gelu(a)
                e = z[:, sl] + h @ W1.transpose(-1, -2) - tgt[:, sl]                 # M_l(z_t) - z_{t+1}
                delta = (e @ W1) * gelu_grad(a); b = uc.shape[1]
                g1, g2 = bounded((eta[:, sl] * e).transpose(-1, -2) @ h / b, (eta[:, sl] * delta).transpose(-1, -2) @ uc / b)
                W1 = W1 - g1; W2 = W2 - g2                                            # Eq. 71 (sum over the chunk)
                outs.append(z[:, sl] + F.gelu(uc @ W2.transpose(-1, -2)) @ W1.transpose(-1, -2))
            z = torch.cat(outs, 1)
        return z


# ------------------------------------------------------------------------------------------------ verification
def _check():
    """Chunked closed-form updates vs an autograd token loop (float64)."""
    torch.manual_seed(0); B, H, b, d, m = 2, 3, 5, 6, 7
    k = torch.randn(B, H, b, d, dtype=torch.float64); t = torch.randn(B, H, b, d, dtype=torch.float64)
    W1 = torch.randn(B, H, d, m, dtype=torch.float64) / 3; W2 = torch.randn(B, H, m, d, dtype=torch.float64) / 3
    rate = torch.rand(B, H, b, 1, dtype=torch.float64)
    g1, g2, _ = mlp_grad(k, t, W1, W2, rate)
    p1, p2 = W1.clone().requires_grad_(True), W2.clone().requires_grad_(True)
    loss = (0.5 * rate[..., 0] * ((mlp_mem(k, p1, p2) - t) ** 2).sum(-1)).sum() / b
    r1, r2 = torch.autograd.grad(loss, [p1, p2])
    print(f'closed-form vs autograd: dW1 {(g1 - r1).abs().max():.2e}, dW2 {(g2 - r2).abs().max():.2e}')
    for mixer in ['hope', 'titans', 'delta', 'none']:
        mod = SelfModifyingTitans(64, heads=4, d_head=16, hidden=32, chunk=2, mixer=mixer)
        x = torch.randn(3, 8, 64); y, s = mod(x)
        if mixer != 'none': y.sum().backward()
        gn = sum(p.grad.abs().sum().item() for p in mod.parameters() if p.grad is not None)
        print(f'mixer {mixer:6s} out {tuple(y.shape)} surprise {tuple(s.shape)} finite {bool(torch.isfinite(y).all())} grad-sum {gn:.3f}')
    cms = ContinuumMemory(64, 128); z = torch.randn(3, 8, 64, requires_grad=True); o = cms(z); o.sum().backward()
    print('CMS', tuple(o.shape), 'finite', bool(torch.isfinite(o).all()), 'W2 grad', cms.W2[0].grad.abs().sum().item() > 0)


if __name__ == '__main__':
    _check()
