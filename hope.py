"""Hope neural learning module (Behrouz et al., Nested Learning, NeurIPS 2025, arXiv 2512.24695, Sec. 7-8) in plain PyTorch,
chunk-parallel, for short token sequences (here: windows of heartbeats).

Two-layer residual memory used everywhere (Eq. 89):  M(z) = z + W1 gelu(W2 z)
Inner objective (L2 regression):  l = 1/2 || M(k) - target ||^2, gradients in closed form:
    a = W2 k, h = gelu(a), e = M(k) - target,  dW1 = e h^T,  delta = (W1^T e) * gelu'(a),  dW2 = delta k^T
Chunk-wise update (Sec. 8.2): all keys/values/rates of a chunk are generated with the memory state at the end of the previous
chunk, the gradient is taken at that state, averaged over the chunk (scaled by the per-token rate eta_t), bounded to Frobenius
norm <= 1, and applied with the retention gate:  W <- (1 - alpha_c) W - G.  Tokens of a chunk read the updated state.

Dual (low-rank) form.  Every in-context update is a sum of b outer products (b = tokens of the chunk), so the per-window weights
are kept as  W = c W0 + A B^T  (W0 the meta-learned initial state shared by the batch, c the accumulated retention, A and B the
stacked update factors, rank <= window length) and are never materialised.  This is exactly equal to the dense update
(hope_dense.py; checked by _check below) and much faster for per-window memories.

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
    """Dense form: z (..., b, d), W1 (..., d, m), W2 (..., m, d) -> z + gelu(z W2^T) W1^T"""
    return z + F.gelu(z @ W2.transpose(-1, -2)) @ W1.transpose(-1, -2)


def mlp_grad(k, target, W1, W2, rate):
    """Dense chunk gradient of 1/2 sum_t rate_t ||M(k_t) - target_t||^2 / b (kept for the closed-form check)."""
    a = k @ W2.transpose(-1, -2); h = F.gelu(a)
    e = k + h @ W1.transpose(-1, -2) - target
    delta = (e @ W1) * gelu_grad(a)
    b = k.shape[-2]
    return (rate * e).transpose(-1, -2) @ h / b, (rate * delta).transpose(-1, -2) @ k / b, e


# ------------------------------------------------------------------------------------------------ low-rank state
def shared_mm(z, W):
    """z (B, *S, b, i) @ W (*S, i, o) without broadcasting W over the batch (matmul broadcasting would copy W B times):
    the batch is folded into the token axis so that one bmm per shared matrix is used."""
    if W.dim() == 2: return z @ W
    S = W.shape[:-2]; Bn, b = z.shape[0], z.shape[-2]
    z = z.expand(Bn, *S, b, z.shape[-1]).movedim(0, -3).reshape(*S, Bn * b, z.shape[-1])
    return (z @ W).reshape(*S, Bn, b, W.shape[-1]).movedim(-3, 0)


def small_mm(x, y):
    """Batched product of tiny per-window matrices (..., p, k) @ (..., k, n) as a broadcast multiply-and-sum: cuBLAS batched
    GEMM pads every tiny matrix to a 128 x 128 tile, which made these products the bottleneck."""
    return (x.unsqueeze(-1) * y.unsqueeze(-3)).sum(-2)


class LR:
    """Per-window matrix W = c * W0 + A @ B^T with W0 (*S, out, in) shared, c (B, *S), A (B, *S, out, R), B (B, *S, in, R)."""

    def __init__(self, W0, c, A=None, B=None):
        self.W0, self.c, self.A, self.B = W0, c, A, B

    def apply(self, z):                         # z (B, *S, b, in) -> z W^T (B, *S, b, out)
        y = self.c[..., None, None] * shared_mm(z, self.W0.transpose(-1, -2))
        return y if self.A is None else y + small_mm(small_mm(z, self.B), self.A.transpose(-1, -2))

    def apply_t(self, e):                       # e (B, *S, b, out) -> e W (B, *S, b, in)
        y = self.c[..., None, None] * shared_mm(e, self.W0)
        return y if self.A is None else y + small_mm(small_mm(e, self.A), self.B.transpose(-1, -2))

    def step(self, keep, GA, GB, scale):
        """W <- keep * W - scale * GA GB^T  (GA (..., out, b), GB (..., in, b))."""
        c = self.c * keep; GA = -scale[..., None, None] * GA
        if self.A is None: return LR(self.W0, c, GA, GB)
        return LR(self.W0, c, torch.cat([keep[..., None, None] * self.A, GA], -1), torch.cat([self.B, GB], -1))

    def select(self, i):                        # one memory of a stack (memories on axis 1 of the state, axis 0 of W0)
        return LR(self.W0[i], self.c[:, i], None if self.A is None else self.A[:, i], None if self.B is None else self.B[:, i])


def lr_norm2(GA, GB):
    """||GA GB^T||_F^2 = sum((GA^T GA) * (GB^T GB))"""
    return (small_mm(GA.transpose(-1, -2), GA) * small_mm(GB.transpose(-1, -2), GB)).sum((-1, -2))


def mem_lr(z, W1, W2):
    """M(z) = z + W1 gelu(W2 z) with low-rank states."""
    return z + W1.apply(F.gelu(W2.apply(z)))


def grad_lr(u, target, W1, W2, rate, z=None):
    """Factors of the bounded chunk gradient of 1/2 sum_t rate_t ||z_t + W1 gelu(W2 u_t) - target_t||^2 / b (z = u if None).
    Returns (A1, B1, A2, B2, scale, e) with dW1 = A1 B1^T, dW2 = A2 B2^T and scale = 1 / max(1, ||(dW1, dW2)||_F)."""
    a = W2.apply(u); h = F.gelu(a)
    e = (u if z is None else z) + W1.apply(h) - target
    delta = W1.apply_t(e) * gelu_grad(a)
    b = u.shape[-2]
    A1, B1 = (rate * e / b).transpose(-1, -2), h.transpose(-1, -2)
    A2, B2 = (rate * delta / b).transpose(-1, -2), u.transpose(-1, -2)
    n = torch.sqrt(lr_norm2(A1, B1) + lr_norm2(A2, B2) + 1e-12)
    return A1, B1, A2, B2, 1.0 / torch.clamp(n, min=1.0), e


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
            # meta-learned initial states of the five memories (outer level), stacked on one axis in the order NAMES
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
            xh = self.heads(self.inp(x), B, L); N = len(self.NAMES)
            one = torch.ones(B, N, H, device=x.device, dtype=x.dtype)
            W1, W2 = LR(self.W1, one), LR(self.W2, one)                                    # states of the five memories
            for c in range(n):
                sl = slice(c * self.C, min(L, (c + 1) * self.C)); xc = xh[:, :, sl]
                G = mem_lr(xc.unsqueeze(1), W1, W2)                                         # Eq. 83 with the previous-chunk states
                k = F.normalize(G[:, 0], dim=-1); v = G[:, 1]
                eta = self.eta_max * torch.sigmoid((G[:, 2] * self.r_eta[None, :, None]).sum(-1) + self.b_eta[None, :, None])      # (B, H, b)
                alpha = self.alpha_max * torch.sigmoid((G[:, 3] * self.r_alpha[None, :, None]).sum(-1)
                                                       + self.b_alpha[None, :, None]).mean(-1)                                    # (B, H)
                vhat = mem_lr(v.unsqueeze(1), W1, W2)                                       # Eq. 84: self-generated values, all memories
                kk = k.unsqueeze(1).expand(-1, N, -1, -1, -1)
                A1, B1, A2, B2, s, e = grad_lr(kk, vhat, W1, W2, eta[:, None, :, :, None])
                keep = (1 - alpha)[:, None, :].expand(-1, N, -1)
                W1, W2 = W1.step(keep, A1, B1, s), W2.step(keep, A2, B2, s)
                surp.append(e[:, 4].norm(dim=-1))
                ys.append(mem_lr(q[:, :, sl], W1.select(4), W2.select(4)))                  # read the updated output memory
        elif self.mixer in ('titans', 'delta'):
            k, v = self.kv(x).view(B, L, 2, H, self.d).permute(2, 0, 3, 1, 4); k = F.normalize(k, dim=-1)
            g = self.gates(x).view(B, L, 2, H).permute(2, 0, 3, 1)
            eta_all = self.eta_max * torch.sigmoid(g[0]); alpha_all = self.alpha_max * torch.sigmoid(g[1])
            one = torch.ones(B, H, device=x.device, dtype=x.dtype)
            if self.mixer == 'titans': W1, W2 = LR(self.W1m, one), LR(self.W2m, one)
            else: W = LR(self.Wl, one)
            for c in range(n):
                sl = slice(c * self.C, min(L, (c + 1) * self.C)); kc, vc = k[:, :, sl], v[:, :, sl]
                eta = eta_all[:, :, sl, None]; keep = 1 - alpha_all[:, :, sl].mean(-1)
                if self.mixer == 'titans':
                    A1, B1, A2, B2, s, e = grad_lr(kc, vc, W1, W2, eta)
                    W1, W2 = W1.step(keep, A1, B1, s), W2.step(keep, A2, B2, s)
                    ys.append(mem_lr(q[:, :, sl], W1, W2))
                else:
                    e = W.apply(kc) - vc                                                      # delta rule
                    GA, GB = (eta * e / kc.shape[-2]).transpose(-1, -2), kc.transpose(-1, -2)
                    s = 1.0 / torch.clamp(torch.sqrt(lr_norm2(GA, GB) + 1e-12), min=1.0)
                    W = W.step(keep, GA, GB, s)
                    ys.append(W.apply(q[:, :, sl]))
                surp.append(e.norm(dim=-1))
        else:
            return torch.zeros_like(x), torch.zeros(B, L, device=x.device)
        y = torch.cat(ys, 2).transpose(1, 2).reshape(B, L, H * self.d)
        return self.out(self.norm(y)), torch.cat(surp, 2).mean(1)


class ContinuumMemory(nn.Module):
    """Sequential CMS: levels with chunk sizes `chunks`, each M_l(z) = z + W1 gelu(W2 LN(z)), updated in context every C_l
    tokens by next-token prediction  1/2 ||M_l(z_t) - z_{t+1}||^2  (the last token has no target).  adaptive=False -> static FFNs."""

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
            if not self.adaptive:
                z = z + F.gelu(self.ln[l](z) @ self.W2[l].t()) @ self.W1[l].t(); continue
            one = torch.ones(B, device=z.device, dtype=z.dtype)
            W1, W2 = LR(self.W1[l], one), LR(self.W2[l], one)
            u = self.ln[l](z)
            tgt = torch.cat([z[:, 1:], z[:, -1:]], 1); valid = torch.ones(B, L, 1, device=z.device, dtype=z.dtype); valid[:, -1] = 0
            eta = self.eta_max * torch.sigmoid(self.rate[l](u)) * valid
            outs = []
            for c in range(math.ceil(L / C)):
                sl = slice(c * C, min(L, (c + 1) * C))
                A1, B1, A2, B2, s, _ = grad_lr(u[:, sl], tgt[:, sl], W1, W2, eta[:, sl], z=z[:, sl])   # M_l(z_t) - z_{t+1}
                W1, W2 = W1.step(one, A1, B1, s), W2.step(one, A2, B2, s)                               # Eq. 71
                outs.append(z[:, sl] + W1.apply(F.gelu(W2.apply(u[:, sl]))))
            z = torch.cat(outs, 1)
        return z


# ------------------------------------------------------------------------------------------------ verification
def _check():
    """(1) closed-form gradient vs autograd; (2) dual (low-rank) modules vs the dense reference implementation (float64)."""
    torch.manual_seed(0); B, H, b, d, m = 2, 3, 5, 6, 7
    k = torch.randn(B, H, b, d, dtype=torch.float64); t = torch.randn(B, H, b, d, dtype=torch.float64)
    W1 = torch.randn(B, H, d, m, dtype=torch.float64) / 3; W2 = torch.randn(B, H, m, d, dtype=torch.float64) / 3
    rate = torch.rand(B, H, b, 1, dtype=torch.float64)
    g1, g2, _ = mlp_grad(k, t, W1, W2, rate)
    p1, p2 = W1.clone().requires_grad_(True), W2.clone().requires_grad_(True)
    loss = (0.5 * rate[..., 0] * ((mlp_mem(k, p1, p2) - t) ** 2).sum(-1)).sum() / b
    r1, r2 = torch.autograd.grad(loss, [p1, p2])
    print(f'closed-form vs autograd: dW1 {(g1 - r1).abs().max():.2e}, dW2 {(g2 - r2).abs().max():.2e}')
    import hope_dense as D
    for mixer in ['hope', 'titans', 'delta']:
        torch.manual_seed(1)
        a = SelfModifyingTitans(64, heads=4, d_head=16, hidden=32, chunk=2, mixer=mixer).double()
        r = D.SelfModifyingTitans(64, heads=4, d_head=16, hidden=32, chunk=2, mixer=mixer).double()
        with torch.no_grad():
            for p in a.parameters(): p.add_(0.05 * torch.randn_like(p))
        r.load_state_dict(a.state_dict())
        x = torch.randn(3, 8, 64, dtype=torch.float64)
        (ya, sa), (yr, sr) = a(x), r(x)
        ga = torch.autograd.grad(ya.square().sum(), list(a.parameters()), allow_unused=True)
        gr = torch.autograd.grad(yr.square().sum(), list(r.parameters()), allow_unused=True)
        print(f'{mixer:6s} dual vs dense: output {(ya - yr).abs().max():.1e}, surprise {(sa - sr).abs().max():.1e}, '
              f'grads {max((u - w).abs().max().item() for u, w in zip(ga, gr) if u is not None):.1e}')
    torch.manual_seed(2)
    a = ContinuumMemory(64, 128).double(); r = D.ContinuumMemory(64, 128).double()
    with torch.no_grad():
        for p in a.parameters(): p.add_(0.05 * torch.randn_like(p))
    r.load_state_dict(a.state_dict()); z = torch.randn(3, 8, 64, dtype=torch.float64)
    ya, yr = a(z), r(z)
    ga = torch.autograd.grad(ya.square().sum(), list(a.parameters())); gr = torch.autograd.grad(yr.square().sum(), list(r.parameters()))
    print(f'CMS    dual vs dense: output {(ya - yr).abs().max():.1e}, grads {max((u - w).abs().max().item() for u, w in zip(ga, gr)):.1e}')


if __name__ == '__main__':
    _check()
