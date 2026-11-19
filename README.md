# HeartNest

HeartNest is an electrocardiogram (ECG) identity encoder with in-context memory. Each heartbeat is a token, and a window of eight beats is processed by two Hope blocks, each made of a self-modifying memory (SMT) and a continuum memory system (CMS). Both memories are updated while the window is read, so the encoder adapts to the person at test time. The network is trained with an ArcFace loss. A new person is enrolled by averaging the embeddings of a short recording, with no retraining.

## Files

| File | Content |
|---|---|
| `00_download.py` | downloads ECG-ID, MIT-BIH Arrhythmia and PTB Diagnostic from the PhysioNet open-data mirror |
| `01_prep.py` | single-lead filtering, R-peak detection, heartbeat tokens, enrolment/probe pools |
| `hope.py` | self-modifying memory and continuum memory system, closed-form in-context gradients, low-rank dual form |
| `hope_dense.py` | reference implementation with explicitly stored weights (used by the gradient check) |
| `heartnest.py` | tokenizer, HeartNest blocks, embedding head, ArcFace, memory variants |
| `train.py` | training and evaluation under the three protocols |

## Requirements

Python 3.13, PyTorch 2.12, NumPy, SciPy and wfdb:

```bash
pip install torch numpy scipy wfdb
```

## Usage

```bash
export HEARTNEST_DATA=data          # where the raw records and heartbeat tokens are stored
python 00_download.py
python 01_prep.py
python hope.py                      # checks the closed-form gradients and the dual form against autograd
python train.py --db ecgiddb --protocol p2 --fold 0 --variant full --seed 0
```

`--db` is `ecgiddb`, `mitdb` or `ptbdb`.

`--protocol` is one of:
- `p1`: closed set.
- `p2`: unseen people, with five person folds (`--fold 0-4`). On ECG-ID it also runs the cross-day test (P3).

`--variant` is one of:
- `full`: HeartNest.
- `nosm`: no self-modification (Titans memory).
- `nocms`: frozen continuum memory.
- `delta`: gated delta rule.
- `nomem`: no in-context memory.

Defaults:
- 30 epochs, AdamW with learning rate 1e-3, batch size 512 windows, 64 windows per training person and epoch.
- Seeds 0–4 were used for `full` and seeds 0–1 for the variants.

Each run writes rank-1 accuracy, equal error rate and the genuine/impostor scores to `$HEARTNEST_OUT` (default `runs/`).

## Data

The three datasets are openly available from PhysioNet:
- ECG-ID: https://physionet.org/content/ecgiddb/1.0.0/
- MIT-BIH Arrhythmia: https://physionet.org/content/mitdb/1.0.0/
- PTB Diagnostic: https://physionet.org/content/ptbdb/1.0.0/

No data are included in this repository.
