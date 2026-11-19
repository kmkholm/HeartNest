"""Train / evaluate HeartNest.

Protocols
  p1  closed-set identification: train on the enrolment pool (first 60% of every record) of all subjects, enrol every subject
      with the mean embedding of its enrolment windows, identify the test-pool windows (last 40% of every record).
  p2  open-set enrolment of unseen people, 5 subject folds: train on 4/5 of the subjects, then enrol the held-out subjects
      without retraining and identify / verify their test-pool windows.  ECG-ID also gets the cross-day test (p3): the held-out
      people are enrolled with every window of their first recording date and probed with the windows of later dates.
Metrics: Rank-1 identification accuracy (window level), equal error rate (probe-vs-template cosine scores).
usage: python train.py --db ecgiddb --protocol p2 --fold 0 --variant full"""
import os, sys, json, time, argparse, datetime, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from heartnest import HeartNest, ArcFace

ROOT = os.environ.get('HEARTNEST_DATA', 'data'); OUT = os.environ.get('HEARTNEST_OUT', 'runs')
L = 8


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--db', required=True); p.add_argument('--protocol', default='p1'); p.add_argument('--fold', type=int, default=0)
    p.add_argument('--variant', default='full'); p.add_argument('--epochs', type=int, default=30); p.add_argument('--per_subject', type=int, default=64)
    p.add_argument('--bs', type=int, default=512); p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=0, help='smoke test: stop after this many steps')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--save_model', action='store_true')
    return p.parse_args()


def windows(rec, pool, pos, records, which_pool, stride):
    """Windows of L consecutive kept beats inside each record and pool; returns {record: (n_w, L) beat indices}."""
    out = {}
    for r in records:
        idx = np.where((rec == r) & (np.isin(pool, which_pool)))[0]
        idx = idx[np.argsort(pos[idx])]
        if len(idx) < L: continue
        st = np.arange(0, len(idx) - L + 1, stride)
        out[r] = np.stack([idx[s:s + L] for s in st])
    return out


def stack(wdict, rec_subj):
    if not wdict: return np.zeros((0, L), int), np.zeros(0, int)
    W = np.concatenate(list(wdict.values())); S = np.concatenate([np.full(len(v), rec_subj[r]) for r, v in wdict.items()])
    return W, S


@torch.no_grad()
def embed(model, beats, rr, W, dev, bs=1024):
    model.eval(); out = []
    for i in range(0, len(W), bs):
        w = torch.as_tensor(W[i:i + bs], device=dev)
        out.append(model(beats[w], rr[w]).float().cpu())
    return torch.cat(out) if out else torch.zeros(0, 128)


def evaluate(E_enrol, S_enrol, E_probe, S_probe):
    subj = np.unique(S_enrol)
    T = F.normalize(torch.stack([E_enrol[torch.as_tensor(S_enrol == s)].mean(0) for s in subj]), dim=-1)
    keep = np.isin(S_probe, subj); E_probe, S_probe = E_probe[torch.as_tensor(keep)], S_probe[keep]
    sc = (E_probe @ T.t()).numpy(); col = np.searchsorted(subj, S_probe)
    rank1 = float((sc.argmax(1) == col).mean())
    per_subj = [float((sc.argmax(1) == col)[S_probe == s].mean()) for s in np.unique(S_probe)]
    gen = sc[np.arange(len(col)), col]; mask = np.ones_like(sc, bool); mask[np.arange(len(col)), col] = False; imp = sc[mask]
    thr = np.unique(np.quantile(np.r_[gen, imp], np.linspace(0, 1, 2001)))
    frr = np.searchsorted(np.sort(gen), thr, side='left') / len(gen)
    far = 1 - np.searchsorted(np.sort(imp), thr, side='left') / len(imp)
    i = np.argmin(np.abs(far - frr)); eer = float((far[i] + frr[i]) / 2)
    return dict(rank1=rank1, balanced_rank1=float(np.mean(per_subj)), eer=eer, n_probe=int(len(S_probe)), n_gallery=int(len(subj)),
                n_probe_subjects=int(len(np.unique(S_probe)))), gen, imp, (sc.argmax(1) == col), S_probe


def main():
    a = parse(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    d = np.load(f'{ROOT}/beats_{a.db}.npz'); meta = json.load(open(f'{ROOT}/meta_{a.db}.json'))
    rec, pool, pos, subj = d['rec'], d['pool'], d['pos'], d['subj']
    beats = torch.as_tensor(d['beats'].astype(np.float32), device=dev); rr = torch.as_tensor(d['rr'], device=dev)
    records = np.unique(rec); rec_subj = {r: subj[rec == r][0] for r in records}
    # subjects usable for evaluation: at least one enrolment window and one test window
    w0, w1 = windows(rec, pool, pos, records, [0], 1), windows(rec, pool, pos, records, [1], 2)
    usable = np.array(sorted({rec_subj[r] for r in w0} & {rec_subj[r] for r in w1}))
    if a.protocol == 'p1':
        train_s, test_s = usable, usable
    else:
        rng = np.random.RandomState(12345); perm = usable[rng.permutation(len(usable))]
        folds = np.array_split(perm, 5); test_s = np.sort(folds[a.fold]); train_s = np.setdiff1d(usable, test_s)
    lab = {s: i for i, s in enumerate(train_s)}
    Wtr, Str = stack({r: v for r, v in w0.items() if rec_subj[r] in lab}, rec_subj)
    Ytr = np.array([lab[s] for s in Str]); by_s = [np.where(Ytr == i)[0] for i in range(len(train_s))]
    model = HeartNest(a.variant, L=L).to(dev); head = ArcFace(128, len(train_s)).to(dev)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps_ep = int(np.ceil(len(train_s) * a.per_subject / a.bs)); total = a.epochs * steps_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05, anneal_strategy='cos')
    t0 = time.time(); step = 0; log = []
    for ep in range(a.epochs):
        model.train(); head.train()
        sel = np.concatenate([np.random.choice(ix, a.per_subject, replace=len(ix) < a.per_subject) for ix in by_s])
        np.random.shuffle(sel); tot = 0.0
        for i in range(0, len(sel), a.bs):
            b = sel[i:i + a.bs]; w = torch.as_tensor(Wtr[b], device=dev); y = torch.as_tensor(Ytr[b], device=dev)
            xb = beats[w]; xb = xb * torch.empty(len(b), 1, 1, device=dev).uniform_(0.8, 1.2)
            xb = xb + 0.03 * xb.std((1, 2), keepdim=True) * torch.randn_like(xb)
            loss, _ = head(model(xb, rr[w]), y)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            tot += loss.item() * len(b); step += 1
            if not np.isfinite(loss.item()): raise RuntimeError('non-finite loss')
            if a.max_steps and step >= a.max_steps: break
        log.append(round(tot / len(sel), 4))
        if a.max_steps and step >= a.max_steps: break
    train_min = (time.time() - t0) / 60
    peak_train_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if dev.type == 'cuda' else None
    # ---- enrolment without retraining + identification / verification
    res = dict(db=a.db, protocol=a.protocol, fold=a.fold, variant=a.variant, seed=a.seed, epochs=a.epochs, steps=step,
               n_train_subjects=int(len(train_s)), n_test_subjects=int(len(test_s)), n_train_windows=int(len(Wtr)),
               train_minutes=round(train_min, 2), peak_train_mb=peak_train_mb, loss=log, time=datetime.datetime.now().isoformat(timespec='seconds'))
    e0, s0 = stack({r: v for r, v in windows(rec, pool, pos, records, [0], 2).items() if rec_subj[r] in set(test_s)}, rec_subj)
    e1, s1 = stack({r: v for r, v in w1.items() if rec_subj[r] in set(test_s)}, rec_subj)
    t1 = time.time(); E0, E1 = embed(model, beats, rr, e0, dev), embed(model, beats, rr, e1, dev)
    res['ms_per_window'] = round(1000 * (time.time() - t1) / max(1, len(e0) + len(e1)), 4)
    m, gen, imp, corr, sp = evaluate(E0, s0, E1, s1); res['main'] = m
    os.makedirs(f'{OUT}/scores', exist_ok=True)
    sfx = '' if a.seed == 0 else f'_s{a.seed}'                      # seed 0 keeps the original file names
    tag = f'{a.db}_{a.protocol}_f{a.fold}_{a.variant}{sfx}'
    rs = np.random.RandomState(0)
    np.savez_compressed(f'{OUT}/scores/{tag}.npz', gen=gen.astype(np.float16), imp=rs.choice(imp, min(len(imp), 200000), replace=False).astype(np.float16),
                        correct=corr, probe_subj=sp)
    if a.db == 'ecgiddb' and a.protocol == 'p2':                       # p3: cross-day enrolment for the held-out people
        dates = {r: datetime.datetime.strptime(meta['record_session'][r], '%d.%m.%Y') for r in records}
        first = {s: min(dates[r] for r in records if rec_subj[r] == s) for s in test_s}
        wall = windows(rec, pool, pos, records, [0, 1], 2)
        en = {r: v for r, v in wall.items() if rec_subj[r] in first and dates[r] == first[rec_subj[r]]}
        pr = {r: v for r, v in wall.items() if rec_subj[r] in first and dates[r] > first[rec_subj[r]]}
        ce, cs = stack(en, rec_subj); pe, ps = stack(pr, rec_subj)
        if len(pe):
            m3, g3, i3, c3, p3 = evaluate(embed(model, beats, rr, ce, dev), cs, embed(model, beats, rr, pe, dev), ps); res['cross_day'] = m3
            np.savez_compressed(f'{OUT}/scores/{a.db}_p3_f{a.fold}_{a.variant}{sfx}.npz', gen=g3.astype(np.float16), imp=i3.astype(np.float16),
                                correct=c3, probe_subj=p3)
    os.makedirs(OUT, exist_ok=True)
    if a.save_model:
        os.makedirs(f'{OUT}/models', exist_ok=True); torch.save(model.state_dict(), f'{OUT}/models/{tag}.pt')
    if not a.max_steps: json.dump(res, open(f'{OUT}/{tag}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'loss'}), flush=True)


if __name__ == '__main__':
    main()
