"""Download ECG-ID, MIT-BIH Arrhythmia and PTB Diagnostic from the PhysioNet open-data S3 mirror.
physionet.org itself was serving at a few kB/s on 2026-10-04, the S3 mirror is identical and fast.
PTB Frank-lead files (.xyz) are skipped: only the 12 standard leads are used."""
import os, re, sys, urllib.request, concurrent.futures as cf

ROOT = os.environ.get('HEARTNEST_DATA', 'data')
BASE = 'https://physionet-open.s3.amazonaws.com'
DBS = {'ecgiddb': 'ecgiddb/1.0.0/', 'mitdb': 'mitdb/1.0.0/', 'ptbdb': 'ptbdb/1.0.0/'}

def list_prefix(prefix):
    keys, token = [], None
    while True:
        url = f'{BASE}/?list-type=2&prefix={prefix}' + (f'&continuation-token={urllib.parse.quote(token)}' if token else '')
        x = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += [(k, int(s)) for k, s in re.findall(r'<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>', x, re.S)]
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', x)
        if not m: return keys
        token = m.group(1)

def fetch(item):
    key, size, dst = item
    if os.path.exists(dst) and os.path.getsize(dst) == size: return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(f'{BASE}/{key}', dst + '.part'); os.replace(dst + '.part', dst); return size
        except Exception as e:
            err = e
    raise RuntimeError(f'{key}: {err}')

import urllib.parse
for db, prefix in DBS.items():
    keys = [(k, s) for k, s in list_prefix(prefix) if not k.endswith('/') and not k.endswith('.xyz')]
    items = [(k, s, os.path.join(ROOT, db, k[len(prefix):])) for k, s in keys]
    print(db, len(items), 'files', round(sum(s for _, s in keys) / 1e6, 1), 'MB', flush=True)
    with cf.ThreadPoolExecutor(16) as ex:
        got = sum(ex.map(fetch, items))
    bad = [d for k, s, d in items if not (os.path.exists(d) and os.path.getsize(d) == s)]
    print(db, 'done', round(got / 1e6, 1), 'MB new, size mismatches:', len(bad), flush=True)
