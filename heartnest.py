"""HeartNest: heartbeat tokenizer -> Hope blocks (self-modifying Titans + Continuum Memory System) -> identity embedding.

Input  beats (B, L, 175) single-lead heartbeats at 250 Hz, rr (B, L, 2) pre/post RR intervals in seconds.
Token  1-D CNN over the beat waveform + RR projection + learned position, dimension D.
Block  x <- x + LocalConv(LN x) ; x <- x + Mixer(LN x) ; x <- CMS(x)      (Hope, Behrouz et al. 2025, Eq. 94-97)
Head   LN -> [mean, max] pooling over beats -> linear -> BN -> L2-normalised embedding (trained with ArcFace).
Enrolment of a new person = mean of the person's window embeddings (no retraining)."""
import torch, torch.nn as nn, torch.nn.functional as F
from hope import SelfModifyingTitans, ContinuumMemory

VARIANTS = {                         # name: (mixer, adaptive CMS)
    'full':     ('hope', True),
    'nosm':     ('titans', True),    # without self-modification: static projections (Titans memory)
    'nocms':    ('hope', False),     # continuum memory frozen in context (static feed-forward levels)
    'delta':    ('delta', True),     # linear gated delta rule (Gated DeltaNet memory)
    'nomem':    ('none', False),     # no in-context memory at all
}


class BeatTokenizer(nn.Module):
    def __init__(self, D=128):
        super().__init__()
        c = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.cnn = nn.Sequential(c(1, 32, 7), nn.MaxPool1d(2), c(32, 64, 5), nn.MaxPool1d(2), c(64, 96, 5), nn.AdaptiveAvgPool1d(8))
        self.proj = nn.Linear(96 * 8, D)
        self.rr = nn.Linear(3, D)

    def forward(self, beats, rr):
        B, L, T = beats.shape
        mu = beats.mean((1, 2), keepdim=True); sd = beats.std((1, 2), keepdim=True) + 1e-3   # window-level amplitude normalisation
        z = self.cnn(((beats - mu) / sd).reshape(B * L, 1, T)).flatten(1)
        rrf = torch.stack([rr[..., 0], rr[..., 1], torch.log(rr[..., 0] / rr[..., 1])], -1)
        return self.proj(z).view(B, L, -1) + self.rr(rrf)


class HeartNestBlock(nn.Module):
    def __init__(self, D, mixer, adaptive, heads=4, d_head=32, hidden=64, chunk=2, cms_chunks=(8, 4, 2), cms_hidden=256):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.conv = nn.Conv1d(D, D, 4, groups=D)                           # local convolution, window 4 (Hope Sec. 8.3)
        self.mixer = SelfModifyingTitans(D, heads, d_head, hidden, chunk, mixer) if mixer != 'none' else None
        self.cms = ContinuumMemory(D, cms_hidden, cms_chunks, adaptive)

    def forward(self, x):
        h = self.ln1(x).transpose(1, 2)
        x = x + self.conv(F.pad(h, (3, 0))).transpose(1, 2)
        s = torch.zeros(x.shape[:2], device=x.device)
        if self.mixer is not None:
            y, s = self.mixer(self.ln2(x)); x = x + y
        return self.cms(x), s


class HeartNest(nn.Module):
    def __init__(self, variant='full', D=128, L=8, blocks=2, emb=128):
        super().__init__()
        mixer, adaptive = VARIANTS[variant]
        self.tok = BeatTokenizer(D)
        self.pos = nn.Parameter(torch.zeros(1, L, D))
        self.blocks = nn.ModuleList([HeartNestBlock(D, mixer, adaptive) for _ in range(blocks)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Linear(2 * D, emb), nn.BatchNorm1d(emb))

    def forward(self, beats, rr, return_surprise=False):
        x = self.tok(beats, rr) + self.pos[:, :beats.shape[1]]
        surprise = []
        for b in self.blocks:
            x, s = b(x); surprise.append(s)
        x = self.ln(x)
        e = F.normalize(self.head(torch.cat([x.mean(1), x.amax(1)], -1)), dim=-1)
        return (e, torch.stack(surprise, 1)) if return_surprise else e


class ArcFace(nn.Module):
    def __init__(self, emb, n_cls, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, emb) * 0.01); self.s, self.m = s, m

    def forward(self, e, y):
        cos = e @ F.normalize(self.W, dim=-1).t()
        theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta.gather(1, y[:, None]) + self.m)
        logits = cos.scatter(1, y[:, None], target) * self.s
        return F.cross_entropy(logits, y), cos


if __name__ == '__main__':
    for v in VARIANTS:
        m = HeartNest(v); b = torch.randn(4, 8, 175); rr = torch.rand(4, 8, 2) * 0.5 + 0.6
        e, s = m(b, rr, True); n = sum(p.numel() for p in m.parameters())
        print(f'{v:6s} emb {tuple(e.shape)} surprise {tuple(s.shape)} params {n / 1e6:.3f}M finite {bool(torch.isfinite(e).all())}')
