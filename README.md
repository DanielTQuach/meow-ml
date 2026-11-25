# meow-ml

Reproducible replication of Ntalampiras et al. (2019), *Automatic Classification of Cat Vocalizations Emitted in Different Contexts* (*Animals*, 9(8), 543), on the public [CatMeows](https://zenodo.org/records/4008297) corpus (440 wav files, 8 kHz).

The public release has 440 clips. The paper used 448 and trained Torch C++ HMMs on MATLAB Modulation Toolbox features. The numbers below are the published comparison points, not a guarantee that this Python pipeline will match them digit for digit.

| Model | Paper accuracy |
| --- | --- |
| DAG-HMM | 95.94% |
| Class-specific GMM-HMM | 80.95% |
| RBF-SVM | 78.51% |

Published DAG-HMM recalls: waiting for food (`F`) 100%, isolation (`I`) 92.59%, brushing (`B`) 95.24%.

## Measured results

The full grid is in `results/metrics.json`. It uses all 440 clips (`F` 92, `I` 221, `B` 127), stratified 10-fold with `random_state=42`, the 116-dimensional MFCC plus temporal-modulation cache, fold-wise z-scoring fit on training frames only, ergodic diagonal GMM-HMMs scored with raw log-likelihood, and a grid of 3–7 states, 1–4 mixtures, and 25 EM iterations.

| Model | This run | Paper |
| --- | --- | --- |
| DAG-HMM | 72.05% ± 4.29 | 95.94% |
| Class-specific GMM-HMM | 74.55% ± 4.89 | 80.95% |
| RBF-SVM | 75.91% ± 5.38 | 78.51% |

DAG-HMM recall on that run: `F` 59.78% (paper 100%), `I` 77.38% (92.59%), `B` 71.65% (95.24%). Fold accuracies ran from 63.6% to 77.3%. The sum of absolute gaps to the three published accuracies is 32.9 points.

The RBF-SVM is 2.6 points under the paper, and the flat class-specific HMM is 6.4 points under. The DAG-HMM is the miss, at 72.05% against 95.94%, and it finished behind the flat HMM. Pairwise inner-validation accuracy was about 70–89%, so the first cut of the DAG is often wrong and cannot be revised.

A later screen of nine settings is in `results/search/summary.json`: MFCC only, modulation only, both feature sets, left-to-right topology, no scaling, per-utterance centering, length-normalized likelihood, and a fixed 0.5 s handle cut in place of the Sohn trim. Each screen held the HMM at 5 states, 2 mixtures, and 10 EM iterations. The closest screen was MFCC after the handle cut. Its full grid, in `results/search/best-full/`, scored DAG-HMM 70.68% ± 11.11, class-specific HMM 70.00% ± 11.12, and RBF-SVM 73.86% ± 5.79, a gap of 40.9 points. Food recall stayed at 59.78%. Left-to-right and modulation-only were the weakest screens. Length normalization did not change the screen decisions. The original full grid above is still the closest run.

## Limitations

- The public Zenodo release has 440 files. The paper trained on 448. The eight missing clips are not in this repository.
- openSMILE left pre-emphasis, cepstral liftering, and the energy formula unpublished. This pipeline uses pre-emphasis 0.97, no lifter, and log frame energy.
- Temporal modulation uses an in-repo gammatone filterbank. The paper used the MATLAB Modulation Toolbox, and those preset parameters were not released.
- The paper’s HMMs were Torch7 C++ (`torch.ch`), which is abandoned and is not PyTorch. No trained models were released. `hmmlearn` is the stand-in.
- The paper does not publish the fold seed, the stratification, or the inner validation fraction. This run uses stratified 10-fold with seed 42 and a 20% inner holdout. The fold standard deviation is about 4 points, which does not account for the 24-point DAG gap.
- Reaching 95.94% would require the missing clips and those unpublished settings. The search above moves accuracy by a few points inside a 56–72% band.

## Setup

```text
python -m pip install -r requirements.txt
python main.py --data-dir data/raw --download
```

`--download` fetches `dataset.zip` from Zenodo into `data/raw` and extracts it. Training uses the CPU. `hmmlearn` and LIBSVM do not run on a GPU.

On a Ryzen 7600X the full 10-fold grid took about 49 minutes with `--n-jobs 6`. The nine-setting screen plus its winning full refit took about 89 minutes. Leave the job count at the number of physical cores. The process pins BLAS to one thread so OpenBLAS does not fight the worker pool.

## Commands

```text
python main.py --smoke
python main.py --data-dir data/raw --cache data/features_cache.npz
python main.py --quick
python main.py --group-by-cat
```

`--smoke` checks the filename parser, silence trimming, the 116-dimensional feature layout, DAG routing, and a tiny model fit. It does not need the corpus.

`--quick` searches fewer HMM settings (`states` in {3, 5}, `mixtures` in {1, 2}, 10 EM iterations) so the wiring can be checked before the full grid.

`--group-by-cat` swaps the stratified recording split for `GroupKFold` on the cat id. Those folds are uneven because a few cats contribute about 50 clips.

Reports are written to `results/metrics.json` and `results/confusion_matrix_dag.csv`.

## Pipeline

1. Parse `C_NNNNN_BB_SS_OOOOO_RXX.wav`. The first character is the label (`F`, `I`, or `B`); `NNNNN` is the cat id; `BB` is `MC` or `EU`.
2. Trim the 0.5 s silence handles with a Sohn et al. (1999) likelihood-ratio detector. An energy and spectral-flatness rule is the fallback. A clip shorter than 150 ms after trimming is expanded around its peak to 300 ms.
3. Extract 52 MFCC features (12 coefficients, log energy, and three derivatives) and 64 temporal-modulation features (16 gammatone channels by 4 bands from 0–32 Hz). The mel matrix is the HTK scale with Slaney area normalization, the same settings as `librosa.filters.mel(..., htk=True, norm="slaney")`, implemented in-repo so extraction does not import Numba. The modulation window is 160 ms, centered on each 30 ms MFCC frame, because a 30 ms window cannot resolve those modulation frequencies.
4. Cache the sequences in `data/features_cache.npz`. Inside each fold, `StandardScaler` is fit on training frames only.
5. Tune pairwise diagonal GMM-HMMs on an inner 20% holdout, place the most accurate pair at the root of the DAG, and compare with a flat 3-class GMM-HMM and an RBF-SVM on utterance mean and standard deviation.

The outer split is `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`.
