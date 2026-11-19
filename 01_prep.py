"""Heartbeat tokens for ECG-ID, MIT-BIH Arrhythmia and PTB Diagnostic (one lead each, identical pipeline).

lead        ECG-ID: 'ECG I' (raw), PTB: 'i', MIT-BIH: channel 0 (MLII in 45 of 48 records)
filter      4th-order Butterworth band-pass 0.5-40 Hz (zero phase), resampled to 250 Hz
R-peaks     wfdb XQRS on the filtered 250 Hz signal, refined to the largest |amplitude| within +-60 ms
token       -0.248 s .. +0.452 s around the R-peak (175 samples) + pre/post RR interval (s)
split       inside every record the first 60% of beats form the enrolment/training pool (0), the last 40% the test pool (1)
subject     ECG-ID person, MIT-BIH record (201 and 202 are the same man), PTB patient
session     ECG-ID acquisition date, PTB record, MIT-BIH record
Output: $HEARTNEST_DATA/beats_<db>.npz + meta_<db>.json"""
import os, re, json, numpy as np, wfdb
from wfdb import processing
from scipy.signal import butter, sosfiltfilt, resample_poly
from multiprocessing import Pool
from math import gcd

ROOT, FS, PRE, POST = os.environ.get('HEARTNEST_DATA', 'data'), 250, 62, 113          # 62 + 113 = 175 samples
SOS = {}


def bandpass(x, fs):
    if fs not in SOS: SOS[fs] = butter(4, [0.5, 40], btype='band', fs=fs, output='sos')
    return sosfiltfilt(SOS[fs], x)


def load(db, rec):
    path = f'{ROOT}/{db}/{rec}'
    if db == 'ecgiddb':
        r = wfdb.rdrecord(path, channel_names=['ECG I'])
        date = re.search(r'ECG date:\s*([0-9.]+)', ' '.join(r.comments)).group(1)
        return r.p_signal[:, 0], r.fs, rec.split('/')[0], date
    if db == 'ptbdb':
        r = wfdb.rdrecord(path, channel_names=['i'])
        return r.p_signal[:, 0], r.fs, rec.split('/')[0], rec
    r = wfdb.rdrecord(path, channels=[0])
    return r.p_signal[:, 0], r.fs, ('201' if rec == '202' else rec), rec


def beats_of(args):
    db, rec = args
    x, fs, subj, sess = load(db, rec)
    x = np.nan_to_num(x)
    x = bandpass(x, fs)
    g = gcd(int(fs), FS); x = resample_poly(x, FS // g, int(fs) // g).astype(np.float64)
    r = processing.xqrs_detect(sig=x, fs=FS, verbose=False)
    w = int(0.06 * FS)
    ref = set()
    for i in r:
        lo, hi = max(0, i - w), min(len(x), i + w + 1); ref.add(int(lo + np.argmax(np.abs(x[lo:hi]))))
    r = np.array(sorted(ref), dtype=int)
    r = r[np.r_[True, np.diff(r) > int(0.2 * FS)]]                  # merge detections closer than 200 ms
    r = r[(r >= PRE) & (r + POST <= len(x))]
    if len(r) < 6: return None
    rr = np.diff(r) / FS
    pre, post = np.r_[np.nan, rr], np.r_[rr, np.nan]
    keep = ~(np.isnan(pre) | np.isnan(post)); keep &= (pre > 0.25) & (pre < 2.0) & (post > 0.25) & (post < 2.0)
    idx = np.where(keep)[0]
    B = np.stack([x[r[i] - PRE:r[i] + POST] for i in idx]).astype(np.float32)
    RR = np.stack([pre[idx], post[idx]], 1).astype(np.float32)
    n = len(idx); pool = (np.arange(n) >= int(0.6 * n)).astype(np.int8)
    return dict(rec=rec, subj=subj, sess=sess, beats=B, rr=RR, pool=pool, pos=np.arange(n), rpos=r[idx].astype(np.int64), dur=len(x) / FS)


def main():
    for db in ['ecgiddb', 'mitdb', 'ptbdb']:
        recs = [l.strip() for l in open(f'{ROOT}/{db}/RECORDS') if l.strip()]
        with Pool(4) as p: out = [o for o in p.map(beats_of, [(db, r) for r in recs]) if o is not None]
        subjects = sorted({o['subj'] for o in out}); sessions = sorted({(o['subj'], o['sess']) for o in out})
        si = {s: i for i, s in enumerate(subjects)}; se = {s: i for i, s in enumerate(sessions)}
        cat = lambda k: np.concatenate([o[k] for o in out])
        n_per = [len(o['pool']) for o in out]
        np.savez_compressed(f'{ROOT}/beats_{db}.npz', beats=cat('beats').astype(np.float16), rr=cat('rr'), pool=cat('pool'),
                            pos=cat('pos'), rpos=cat('rpos'), subj=np.repeat([si[o['subj']] for o in out], n_per),
                            rec=np.repeat(np.arange(len(out)), n_per), sess=np.repeat([se[(o['subj'], o['sess'])] for o in out], n_per))
        meta = dict(db=db, fs=FS, token=[-PRE / FS, POST / FS], subjects=subjects, records=[o['rec'] for o in out],
                    record_subject=[o['subj'] for o in out], record_session=[o['sess'] for o in out],
                    record_seconds=[round(o['dur'], 1) for o in out], record_beats=n_per,
                    n_records_in=len(recs), n_records_used=len(out), n_beats=int(sum(n_per)))
        json.dump(meta, open(f'{ROOT}/meta_{db}.json', 'w'), indent=1)
        nb = np.array(n_per)
        print(f'{db}: {len(out)}/{len(recs)} records, {len(subjects)} subjects, {len(sessions)} sessions, '
              f'{nb.sum()} beats (per record median {int(np.median(nb))}, min {nb.min()})', flush=True)


if __name__ == '__main__':
    main()
