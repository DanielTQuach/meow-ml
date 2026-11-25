"""Pairwise GMM-HMMs, a topologically ordered DAG-HMM, and the baselines.

Hidden Markov models are ``hmmlearn`` diagonal ``GMMHMM``s. Mixture weights
are included (``init_params='stmcw'`` and ``params='stmcw'``). hmmlearn 0.3
does not allocate ``weights_`` unless ``w`` is present, and EM has to update
those weights for a mixture emission. K-means initialization is capped at 50
iterations, matching the paper, by wrapping the ``KMeans`` objects hmmlearn
constructs internally.

The three-class problem is split into the pairs (F, I), (F, B), and (I, B).
On an inner stratified holdout of the training fold, each pair is tuned over
states and mixture components. The most accurate pair becomes the root and
eliminates the losing class. The child node then compares the winner with the
remaining class.

Baselines are a flat three-model GMM-HMM and an RBF-SVM on the mean and
standard deviation of each utterance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GMMHMM
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

LABELS: tuple[str, ...] = ("F", "I", "B")
PAIRS: tuple[tuple[str, str], ...] = (("F", "I"), ("F", "B"), ("I", "B"))


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameter grids and training limits."""

    state_grid: tuple[int, ...] = (3, 4, 5, 6, 7)
    mix_grid: tuple[int, ...] = (1, 2, 3, 4)
    n_iter: int = 25
    tol: float = 1e-3
    min_covar: float = 1e-3
    kmeans_max_iter: int = 50
    inner_val_fraction: float = 0.2
    n_jobs: int = 6
    seed: int = 42
    svm_C: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svm_gamma: tuple[float | str, ...] = ("scale", 0.01, 0.001)
    topology: str = "ergodic"
    length_normalize: bool = False


def quick_model_config(n_jobs: int = 6, seed: int = 42) -> ModelConfig:
    """Smaller grid used by ``--quick`` to check the pipeline."""

    return ModelConfig(
        state_grid=(3, 5),
        mix_grid=(1, 2),
        n_iter=10,
        n_jobs=n_jobs,
        seed=seed,
    )


class PaperGMMHMM(GMMHMM):
    """Diagonal GMM-HMM whose k-means initialization stops after 50 iterations."""

    kmeans_max_iter = 50

    def _init(self, X, lengths=None):
        import hmmlearn.hmm as hmm_module

        original = hmm_module.cluster.KMeans
        max_iter = self.kmeans_max_iter

        def capped_kmeans(*args, **kwargs):
            kwargs["max_iter"] = max_iter
            return original(*args, **kwargs)

        hmm_module.cluster.KMeans = capped_kmeans
        try:
            super()._init(X, lengths)
        finally:
            hmm_module.cluster.KMeans = original


def model_seed(fold_seed: int, n_components: int, n_mix: int, class_index: int) -> int:
    """Deterministic seed for one Gaussian-mixture HMM."""

    value = fold_seed + n_components * 17 + n_mix * 101 + class_index * 1009
    return int(value % (2**31 - 1))


def safe_score(
    model: GMMHMM,
    sequence: NDArray[np.floating],
    length_normalize: bool = False,
) -> float:
    """Log-likelihood of one sequence, or ``-inf`` when scoring fails."""

    try:
        features = np.asarray(sequence, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] == 0:
            return float("-inf")
        score = float(model.score(features))
        if length_normalize:
            score /= float(features.shape[0])
    except Exception:
        return float("-inf")
    if not np.isfinite(score):
        return float("-inf")
    return score


def _force_left_to_right(model: PaperGMMHMM) -> None:
    """Allow only a self-loop and a step into the next state, starting at state 0."""

    states = int(model.n_components)
    transmat = np.zeros((states, states), dtype=np.float64)
    for index in range(states):
        if index + 1 < states:
            transmat[index, index] = 0.5
            transmat[index, index + 1] = 0.5
        else:
            transmat[index, index] = 1.0
    startprob = np.zeros(states, dtype=np.float64)
    startprob[0] = 1.0
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.startprob_prior = startprob.copy()
    model.transmat_prior = (transmat > 0.0).astype(np.float64)


def fit_gmmhmm(
    sequences: list[NDArray[np.floating]],
    n_components: int,
    n_mix: int,
    n_iter: int,
    tol: float,
    min_covar: float,
    seed: int,
    topology: str = "ergodic",
) -> PaperGMMHMM:
    """Fit one class-conditional model on already scaled sequences."""

    usable = [np.asarray(sequence, dtype=np.float64) for sequence in sequences if len(sequence)]
    if not usable:
        raise ValueError("Cannot fit a GMM-HMM without sequences.")
    left_to_right = topology == "left-to-right"
    model = PaperGMMHMM(
        n_components=n_components,
        n_mix=n_mix,
        covariance_type="diag",
        min_covar=min_covar,
        n_iter=n_iter,
        tol=tol,
        init_params="mcw" if left_to_right else "stmcw",
        params="stmcw",
        random_state=seed,
        implementation="log",
        verbose=False,
    )
    model.kmeans_max_iter = 50
    if left_to_right:
        _force_left_to_right(model)
    stacked = np.vstack(usable)
    lengths = [int(sequence.shape[0]) for sequence in usable]
    np.random.seed(seed)
    with threadpool_limits(limits=1):
        model.fit(stacked, lengths)
    return model


def child_pair(root: tuple[str, str], winner: str) -> tuple[str, str]:
    """Pair that compares the root winner with the class the root did not see."""

    if winner not in root:
        raise ValueError(f"Winner {winner} is not in the root pair {root}.")
    remaining = next(label for label in LABELS if label not in root)
    wanted = {winner, remaining}
    for pair in PAIRS:
        if set(pair) == wanted:
            return pair
    raise RuntimeError(f"No child pair for root {root} and winner {winner}.")


def _as_str_array(labels: NDArray[np.generic] | list[str]) -> NDArray[np.str_]:
    return np.asarray(labels, dtype=str)


def inner_split(
    labels: NDArray[np.str_],
    fraction: float,
    seed: int,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Stratified holdout. Falls back to one example per class when necessary."""

    labels = _as_str_array(labels)
    counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    if min(counts.values()) >= 2 and len(labels) >= 4:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=fraction, random_state=seed
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
        return train_idx.astype(int), val_idx.astype(int)

    train: list[int] = []
    val: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) == 1:
            train.append(int(indices[0]))
            continue
        val.append(int(indices[0]))
        train.extend(int(index) for index in indices[1:])
    if not val:
        val.append(train.pop())
    return np.asarray(train, dtype=int), np.asarray(val, dtype=int)


def _score_config(
    sequences: list[NDArray[np.float64]],
    labels: NDArray[np.str_],
    train_idx: NDArray[np.int_],
    val_idx: NDArray[np.int_],
    classes: tuple[str, ...],
    n_components: int,
    n_mix: int,
    n_iter: int,
    tol: float,
    min_covar: float,
    fold_seed: int,
    topology: str,
    length_normalize: bool,
) -> dict[str, object]:
    """Train one grid cell and return its inner-validation accuracy."""

    result: dict[str, object] = {
        "accuracy": -1.0,
        "n_components": n_components,
        "n_mix": n_mix,
        "error": "",
    }
    try:
        train_labels = labels[train_idx]
        for class_index, label in enumerate(classes):
            n_train = int(np.sum(train_labels == label))
            if n_train < n_components:
                result["error"] = (
                    f"{label} has {n_train} training sequences for "
                    f"{n_components} states"
                )
                return result
        models: dict[str, PaperGMMHMM] = {}
        for class_index, label in enumerate(classes):
            chosen = train_idx[train_labels == label]
            class_sequences = [sequences[int(index)] for index in chosen]
            models[label] = fit_gmmhmm(
                class_sequences,
                n_components,
                n_mix,
                n_iter,
                tol,
                min_covar,
                model_seed(fold_seed, n_components, n_mix, class_index),
                topology,
            )
        correct = 0
        total = 0
        for index in val_idx:
            label = str(labels[int(index)])
            if label not in classes:
                continue
            scores = [
                safe_score(models[name], sequences[int(index)], length_normalize)
                for name in classes
            ]
            predicted = classes[int(np.argmax(scores))]
            correct += int(predicted == label)
            total += 1
        if total == 0:
            result["error"] = "validation split has no sequences for this task"
            return result
        result["accuracy"] = correct / total
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _select_best(results: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in results if float(item["accuracy"]) >= 0.0]
    if not valid:
        details = [str(item["error"]) for item in results if item.get("error")]
        message = details[0] if details else "every grid cell failed"
        raise RuntimeError(f"HMM grid search failed: {message}")
    return max(
        valid,
        key=lambda item: (
            float(item["accuracy"]),
            -int(item["n_components"]),
            -int(item["n_mix"]),
        ),
    )


def _run_grid(
    sequences: list[NDArray[np.float64]],
    labels: NDArray[np.str_],
    train_idx: NDArray[np.int_],
    val_idx: NDArray[np.int_],
    classes: tuple[str, ...],
    config: ModelConfig,
    fold_seed: int,
) -> dict[str, object]:
    tasks = []
    for n_components in config.state_grid:
        for n_mix in config.mix_grid:
            tasks.append(
                delayed(_score_config)(
                    sequences,
                    labels,
                    train_idx,
                    val_idx,
                    classes,
                    int(n_components),
                    int(n_mix),
                    config.n_iter,
                    config.tol,
                    config.min_covar,
                    fold_seed,
                    config.topology,
                    config.length_normalize,
                )
            )
    if not tasks:
        raise ValueError("The state and mixture grids are empty.")
    results = Parallel(n_jobs=config.n_jobs)(tasks)
    return _select_best(list(results))


def _fit_classes(
    sequences: list[NDArray[np.float64]],
    labels: NDArray[np.str_],
    classes: tuple[str, ...],
    n_components: int,
    n_mix: int,
    config: ModelConfig,
    fold_seed: int,
) -> dict[str, PaperGMMHMM]:
    jobs = []
    for class_index, label in enumerate(classes):
        class_sequences = [
            sequences[int(index)]
            for index, sample_label in enumerate(labels)
            if sample_label == label
        ]
        jobs.append(
            delayed(fit_gmmhmm)(
                class_sequences,
                n_components,
                n_mix,
                config.n_iter,
                config.tol,
                config.min_covar,
                model_seed(fold_seed, n_components, n_mix, class_index),
                config.topology,
            )
        )
    fitted = Parallel(n_jobs=min(config.n_jobs, len(jobs)))(jobs)
    return {label: model for label, model in zip(classes, fitted)}


def _predict_among(
    models: dict[str, PaperGMMHMM],
    sequence: NDArray[np.floating],
    classes: tuple[str, ...],
    length_normalize: bool,
) -> str:
    scores = [
        safe_score(models[label], sequence, length_normalize) for label in classes
    ]
    return classes[int(np.argmax(scores))]


class DAGHMM:
    """Three pairwise GMM-HMM nodes ordered by inner-validation accuracy."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.nodes: dict[tuple[str, str], dict[str, PaperGMMHMM]] = {}
        self.root: tuple[str, str] | None = None
        self.pair_scores: dict[tuple[str, str], float] = {}
        self.pair_params: dict[tuple[str, str], tuple[int, int]] = {}

    def fit(
        self,
        sequences: list[NDArray[np.floating]],
        labels: NDArray[np.generic] | list[str],
        fold_index: int = 0,
    ) -> DAGHMM:
        features = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
        encoded = _as_str_array(labels)
        fold_seed = self.config.seed + int(fold_index)
        train_idx, val_idx = inner_split(
            encoded, self.config.inner_val_fraction, fold_seed
        )
        for pair in PAIRS:
            best = _run_grid(
                features, encoded, train_idx, val_idx, pair, self.config, fold_seed
            )
            self.pair_scores[pair] = float(best["accuracy"])
            self.pair_params[pair] = (int(best["n_components"]), int(best["n_mix"]))
        self.root = max(
            PAIRS,
            key=lambda pair: (
                self.pair_scores[pair],
                -self.pair_params[pair][0],
                -self.pair_params[pair][1],
            ),
        )
        for pair in PAIRS:
            n_components, n_mix = self.pair_params[pair]
            self.nodes[pair] = _fit_classes(
                features, encoded, pair, n_components, n_mix, self.config, fold_seed
            )
        return self

    def predict_one(self, sequence: NDArray[np.floating]) -> str:
        if self.root is None:
            raise RuntimeError("DAGHMM.predict_one called before fit.")
        normalize = self.config.length_normalize
        winner = _predict_among(self.nodes[self.root], sequence, self.root, normalize)
        child = child_pair(self.root, winner)
        return _predict_among(self.nodes[child], sequence, child, normalize)

    def predict(self, sequences: list[NDArray[np.floating]]) -> NDArray[np.str_]:
        return np.asarray([self.predict_one(sequence) for sequence in sequences], dtype=str)


class ClassSpecificHMM:
    """Three GMM-HMMs. The class with the highest log-likelihood wins."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.models: dict[str, PaperGMMHMM] = {}
        self.n_components: int | None = None
        self.n_mix: int | None = None
        self.val_accuracy: float | None = None

    def fit(
        self,
        sequences: list[NDArray[np.floating]],
        labels: NDArray[np.generic] | list[str],
        fold_index: int = 0,
    ) -> ClassSpecificHMM:
        features = [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
        encoded = _as_str_array(labels)
        fold_seed = self.config.seed + int(fold_index)
        train_idx, val_idx = inner_split(
            encoded, self.config.inner_val_fraction, fold_seed
        )
        best = _run_grid(
            features, encoded, train_idx, val_idx, LABELS, self.config, fold_seed
        )
        self.n_components = int(best["n_components"])
        self.n_mix = int(best["n_mix"])
        self.val_accuracy = float(best["accuracy"])
        self.models = _fit_classes(
            features,
            encoded,
            LABELS,
            self.n_components,
            self.n_mix,
            self.config,
            fold_seed,
        )
        return self

    def predict(self, sequences: list[NDArray[np.floating]]) -> NDArray[np.str_]:
        if not self.models:
            raise RuntimeError("ClassSpecificHMM.predict called before fit.")
        normalize = self.config.length_normalize
        predicted = [
            _predict_among(self.models, sequence, LABELS, normalize) for sequence in sequences
        ]
        return np.asarray(predicted, dtype=str)


def pool_utterance(sequence: NDArray[np.floating]) -> NDArray[np.float64]:
    """Mean and standard deviation of a frame sequence."""

    features = np.asarray(sequence, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("Cannot pool an empty sequence.")
    return np.concatenate([features.mean(axis=0), features.std(axis=0)])


class UtteranceRbfSVM:
    """RBF-SVM on utterance-level mean and standard deviation."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.model: SVC | None = None
        self.best_params_: dict[str, object] | None = None

    def fit(
        self,
        sequences: list[NDArray[np.floating]],
        labels: NDArray[np.generic] | list[str],
        fold_index: int = 0,
    ) -> UtteranceRbfSVM:
        del fold_index
        features = np.vstack([pool_utterance(sequence) for sequence in sequences])
        encoded = _as_str_array(labels)
        counts = np.unique(encoded, return_counts=True)[1]
        min_count = int(np.min(counts))
        base = SVC(kernel="rbf", random_state=self.config.seed)
        if min_count < 2:
            base.set_params(C=1.0, gamma="scale")
            with threadpool_limits(limits=1):
                base.fit(features, encoded)
            self.model = base
            self.best_params_ = {"C": 1.0, "gamma": "scale"}
            return self
        n_splits = max(2, min(5, min_count))
        grid = GridSearchCV(
            base,
            param_grid={
                "C": list(self.config.svm_C),
                "gamma": list(self.config.svm_gamma),
            },
            cv=n_splits,
            n_jobs=1,
            refit=True,
        )
        with threadpool_limits(limits=1):
            grid.fit(features, encoded)
        self.model = grid.best_estimator_
        self.best_params_ = dict(grid.best_params_)
        return self

    def predict(self, sequences: list[NDArray[np.floating]]) -> NDArray[np.str_]:
        if self.model is None:
            raise RuntimeError("UtteranceRbfSVM.predict called before fit.")
        features = np.vstack([pool_utterance(sequence) for sequence in sequences])
        return np.asarray(self.model.predict(features), dtype=str)
