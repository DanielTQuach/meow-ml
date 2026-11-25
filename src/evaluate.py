"""Cross-validation, reporting, and the synthetic smoke test.

The outer split is stratified 10-fold on recordings, with a fixed seed, and
the same indices are used for the DAG-HMM, the class-specific HMM, and the
RBF-SVM. ``--group-by-cat`` replaces that split with ``GroupKFold`` on the
cat id. ``--one-cat`` trains each three-context cat alone and scores the
other cats of the same breed and of the other breed. The scaler for a fold
is fit on that fold's training frames only.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.dataset import (
    Recording,
    VadConfig,
    load_audio,
    parse_filename,
    trim_silence,
)
from src.features import (
    N_FEATURES,
    FeatureConfig,
    FeatureExtractor,
    load_feature_cache,
    pipeline_hash,
    save_feature_cache,
)
from src.models import (
    LABELS,
    ClassSpecificHMM,
    DAGHMM,
    ModelConfig,
    UtteranceRbfSVM,
    child_pair,
)

PAPER_ACCURACY = {
    "dag_hmm": 95.94,
    "class_specific_hmm": 80.95,
    "rbf_svm": 78.51,
}
PAPER_RECALL = {"F": 100.0, "I": 92.59, "B": 95.24}


MFCC_DIM = 52


def slice_feature_view(
    sequences: list[NDArray[np.floating]],
    feature_view: str,
) -> list[NDArray[np.float64]]:
    """Keep both streams, the 52 MFCCs, or the 64 modulation features."""

    if feature_view == "both":
        return [np.asarray(sequence, dtype=np.float64) for sequence in sequences]
    if feature_view == "mfcc":
        return [np.asarray(sequence[:, :MFCC_DIM], dtype=np.float64) for sequence in sequences]
    if feature_view == "tm":
        return [np.asarray(sequence[:, MFCC_DIM:], dtype=np.float64) for sequence in sequences]
    raise ValueError(f"Unknown feature view {feature_view!r}.")


def center_utterances(
    sequences: list[NDArray[np.floating]],
) -> list[NDArray[np.float64]]:
    """Subtract each clip's own frame mean."""

    centered: list[NDArray[np.float64]] = []
    for sequence in sequences:
        data = np.asarray(sequence, dtype=np.float64)
        centered.append(data - data.mean(axis=0, keepdims=True))
    return centered


def scale_sequences(
    train: list[NDArray[np.floating]],
    test: list[NDArray[np.floating]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Z-score frames using a scaler fit on the training sequences only."""

    scaler = StandardScaler()
    scaler.fit(np.vstack(train))
    train_scaled = [np.asarray(scaler.transform(sequence), dtype=np.float64) for sequence in train]
    test_scaled = [np.asarray(scaler.transform(sequence), dtype=np.float64) for sequence in test]
    return train_scaled, test_scaled


def make_splits(
    labels: NDArray[np.str_],
    groups: NDArray[np.str_] | None,
    group_by_cat: bool,
    seed: int,
) -> list[tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """Build the outer folds. Grouped folds are uneven for prolific cats."""

    placeholder = np.zeros(len(labels))
    if group_by_cat:
        if groups is None:
            raise ValueError("group_by_cat requires cat ids.")
        n_groups = len(np.unique(groups))
        n_splits = min(10, n_groups)
        if n_splits < 2:
            raise ValueError("GroupKFold needs at least two cats.")
        print(
            "GroupKFold folds are uneven because some cats contribute about 50 clips."
        )
        splitter = GroupKFold(n_splits=n_splits)
        return [
            (train.astype(int), test.astype(int))
            for train, test in splitter.split(placeholder, labels, groups)
        ]
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    return [
        (train.astype(int), test.astype(int))
        for train, test in splitter.split(placeholder, labels)
    ]


def _accuracy(predicted: NDArray[np.str_], truth: NDArray[np.str_]) -> float:
    if len(predicted) == 0:
        return float("nan")
    return float(np.mean(predicted == truth))


def _summarize(scores: list[float]) -> dict[str, float | list[float]]:
    values = np.asarray(scores, dtype=np.float64)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "folds": [float(score) for score in values],
    }


def _confusion_percent(
    truth: list[str], predicted: list[str]
) -> NDArray[np.float64]:
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.float64)
    index = {label: position for position, label in enumerate(LABELS)}
    for true_label, pred_label in zip(truth, predicted):
        matrix[index[true_label], index[pred_label]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return 100.0 * matrix / row_sums


def _print_summary(metrics: dict[str, object]) -> None:
    names = (
        ("dag_hmm", "DAG-HMM"),
        ("class_specific_hmm", "Class-specific HMM"),
        ("rbf_svm", "RBF-SVM"),
    )
    print()
    for key, title in names:
        block = metrics[key]
        assert isinstance(block, dict)
        mean = 100.0 * float(block["mean"])
        std = 100.0 * float(block["std"])
        paper = PAPER_ACCURACY[key]
        print(f"{title} accuracy: {mean:.2f}% ± {std:.2f}%  (paper {paper:.2f}%)")

    print("\nDAG-HMM confusion matrix (row-normalized %):")
    header = "        " + " ".join(f"{label:>8}" for label in LABELS)
    print(header)
    matrix = np.asarray(metrics["confusion_matrix_dag_percent"], dtype=np.float64)
    for label, row in zip(LABELS, matrix):
        cells = " ".join(f"{value:8.2f}" for value in row)
        print(f"{label:>8} {cells}")

    print("\nPer-class recall versus the paper:")
    recall = metrics["recall"]
    assert isinstance(recall, dict)
    for label in LABELS:
        print(
            f"  {label}: {float(recall[label]):.2f}%  "
            f"(paper {PAPER_RECALL[label]:.2f}%)"
        )


def _write_reports(metrics: dict[str, object], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    csv_path = results_dir / "confusion_matrix_dag.csv"
    matrix = np.asarray(metrics["confusion_matrix_dag_percent"], dtype=np.float64)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(LABELS))
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *(f"{value:.4f}" for value in row)])
    print(f"\nWrote {json_path} and {csv_path}")


def evaluate(
    sequences: list[NDArray[np.floating]],
    recordings: list[Recording],
    results_dir: Path,
    group_by_cat: bool = False,
    model_config: ModelConfig | None = None,
    feature_view: str = "both",
    scaler_mode: str = "fold",
) -> dict[str, object]:
    """Run the outer cross-validation and write the report files."""

    config = model_config or ModelConfig()
    prepared = slice_feature_view(sequences, feature_view)
    if scaler_mode == "utterance":
        prepared = center_utterances(prepared)
    elif scaler_mode not in {"fold", "none"}:
        raise ValueError(f"Unknown scaler mode {scaler_mode!r}.")
    labels = np.asarray([recording.label for recording in recordings], dtype=str)
    groups = np.asarray([recording.cat_id for recording in recordings], dtype=str)
    splits = make_splits(labels, groups, group_by_cat, config.seed)
    dag_scores: list[float] = []
    hmm_scores: list[float] = []
    svm_scores: list[float] = []
    dag_truth: list[str] = []
    dag_predicted: list[str] = []
    fold_details: list[dict[str, object]] = []

    for fold_index, (train_idx, test_idx) in enumerate(tqdm(splits, desc="folds")):
        train_sequences = [prepared[int(index)] for index in train_idx]
        test_sequences = [prepared[int(index)] for index in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        if scaler_mode == "fold":
            train_scaled, test_scaled = scale_sequences(train_sequences, test_sequences)
        else:
            train_scaled, test_scaled = train_sequences, test_sequences

        dag = DAGHMM(config).fit(train_scaled, train_labels, fold_index=fold_index)
        class_hmm = ClassSpecificHMM(config).fit(
            train_scaled, train_labels, fold_index=fold_index
        )
        svm = UtteranceRbfSVM(config).fit(train_scaled, train_labels, fold_index=fold_index)

        dag_pred = dag.predict(test_scaled)
        hmm_pred = class_hmm.predict(test_scaled)
        svm_pred = svm.predict(test_scaled)
        dag_acc = _accuracy(dag_pred, test_labels)
        hmm_acc = _accuracy(hmm_pred, test_labels)
        svm_acc = _accuracy(svm_pred, test_labels)
        dag_scores.append(dag_acc)
        hmm_scores.append(hmm_acc)
        svm_scores.append(svm_acc)
        dag_truth.extend(str(label) for label in test_labels)
        dag_predicted.extend(str(label) for label in dag_pred)

        root = dag.root
        root_name = "none" if root is None else f"{root[0]}-vs-{root[1]}"
        print(
            f"Fold {fold_index + 1}/{len(splits)}: "
            f"DAG {dag_acc:.2%}  HMM {hmm_acc:.2%}  SVM {svm_acc:.2%}  "
            f"root {root_name}"
        )
        fold_details.append(
            {
                "fold": fold_index,
                "dag_hmm": dag_acc,
                "class_specific_hmm": hmm_acc,
                "rbf_svm": svm_acc,
                "root_pair": list(root) if root is not None else None,
                "pair_validation_accuracy": {
                    f"{pair[0]}-vs-{pair[1]}": score
                    for pair, score in dag.pair_scores.items()
                },
                "pair_params": {
                    f"{pair[0]}-vs-{pair[1]}": list(params)
                    for pair, params in dag.pair_params.items()
                },
                "class_specific_params": [
                    class_hmm.n_components,
                    class_hmm.n_mix,
                ],
                "svm_params": svm.best_params_,
            }
        )

    confusion = _confusion_percent(dag_truth, dag_predicted)
    recall = {
        label: float(confusion[position, position])
        for position, label in enumerate(LABELS)
    }
    metrics: dict[str, object] = {
        "group_by_cat": group_by_cat,
        "feature_view": feature_view,
        "scaler_mode": scaler_mode,
        "topology": config.topology,
        "length_normalize": config.length_normalize,
        "n_recordings": len(recordings),
        "dag_hmm": _summarize(dag_scores),
        "class_specific_hmm": _summarize(hmm_scores),
        "rbf_svm": _summarize(svm_scores),
        "confusion_matrix_dag_percent": confusion.round(4).tolist(),
        "labels": list(LABELS),
        "recall": recall,
        "paper_accuracy": PAPER_ACCURACY,
        "paper_recall": PAPER_RECALL,
        "folds": fold_details,
    }
    _print_summary(metrics)
    _write_reports(metrics, Path(results_dir))
    return metrics


def prepare_features(
    recordings: list[Recording],
    cache_path: Path,
    feature_config: FeatureConfig | None = None,
    vad_config: VadConfig | None = None,
) -> list[NDArray[np.float64]]:
    """Load the feature cache, or trim, extract, and store a new one."""

    features_config = feature_config or FeatureConfig()
    activity_config = vad_config or VadConfig()
    config_hash = pipeline_hash(features_config, activity_config)
    cached = load_feature_cache(cache_path, recordings, config_hash)
    if cached is not None:
        print(f"Loaded {len(cached)} cached sequences from {cache_path}")
        return cached

    extractor = FeatureExtractor(features_config)
    sequences: list[NDArray[np.float64]] = []
    for recording in tqdm(recordings, desc="features"):
        audio = load_audio(recording.path, target_sr=activity_config.sample_rate)
        trimmed = trim_silence(audio, activity_config)
        sequences.append(extractor.transform(trimmed))
    save_feature_cache(cache_path, sequences, recordings, config_hash)
    print(f"Saved features to {cache_path}")
    return sequences


def run_smoke() -> None:
    """Check parsing, VAD, feature shape, routing, and a tiny model fit."""

    parsed = parse_filename("B_ANI01_MC_FN_SIM01_101.wav")
    if parsed is None or parsed.label != "B" or parsed.cat_id != "ANI01" or parsed.breed != "MC":
        raise AssertionError("Filename parser failed on a valid CatMeows name.")
    if parsed.sex != "FN" or parsed.owner != "SIM01" or parsed.session != "1" or parsed.index != "01":
        raise AssertionError("Filename parser dropped sex, owner, or session fields.")
    if parse_filename("not_a_meow.wav") is not None:
        raise AssertionError("Filename parser accepted a non-CatMeows name.")
    session4 = parse_filename("I_BLE01_EU_FN_DEL01_401.wav")
    if session4 is None or session4.label != "I" or session4.session != "4":
        raise AssertionError("Session 4 CatMeows names must be kept.")

    sample_rate = 8000
    rng = np.random.default_rng(42)
    silence = 0.02 * rng.normal(size=int(0.5 * sample_rate))
    times = np.arange(int(0.4 * sample_rate)) / sample_rate
    tone = np.sin(2.0 * np.pi * 700.0 * times)
    waveform = np.concatenate([silence, tone, silence])
    trimmed = trim_silence(waveform, VadConfig())
    if len(trimmed) == 0 or len(trimmed) >= len(waveform):
        raise AssertionError("VAD did not remove the silence handles.")
    if float(np.max(np.abs(trimmed))) < 0.5:
        raise AssertionError("VAD removed the vocalization peak.")

    silent = trim_silence(np.zeros(sample_rate, dtype=np.float64), VadConfig())
    if len(silent) == 0:
        raise AssertionError("VAD returned an empty clip.")
    if len(silent) < int(0.30 * sample_rate):
        raise AssertionError("Over-trimmed silence was not expanded to 300 ms.")

    features = FeatureExtractor().transform(trimmed)
    if features.shape[1] != N_FEATURES or features.shape[0] < 1:
        raise AssertionError(f"Unexpected feature shape {features.shape}.")
    if not np.all(np.isfinite(features)):
        raise AssertionError("Feature matrix contains non-finite values.")

    expected_children = {
        (("F", "I"), "F"): ("F", "B"),
        (("F", "I"), "I"): ("I", "B"),
        (("F", "B"), "F"): ("F", "I"),
        (("F", "B"), "B"): ("I", "B"),
        (("I", "B"), "I"): ("F", "I"),
        (("I", "B"), "B"): ("F", "B"),
    }
    for (root, winner), child in expected_children.items():
        if child_pair(root, winner) != child:
            raise AssertionError(f"Routing {root} via {winner} did not reach {child}.")

    config = ModelConfig(
        state_grid=(2,),
        mix_grid=(1,),
        n_iter=3,
        n_jobs=1,
        seed=42,
        inner_val_fraction=0.25,
    )
    sequences: list[NDArray[np.float64]] = []
    labels: list[str] = []
    for class_index, label in enumerate(LABELS):
        mean = np.full(8, float(class_index * 4))
        for _ in range(8):
            sequences.append(mean + 0.05 * rng.normal(size=(24, 8)))
            labels.append(label)
    dag = DAGHMM(config).fit(sequences, labels, fold_index=0)
    class_hmm = ClassSpecificHMM(config).fit(sequences, labels, fold_index=0)
    svm = UtteranceRbfSVM(config).fit(sequences, labels, fold_index=0)
    for model in (dag, class_hmm, svm):
        predicted = model.predict(sequences)
        if len(predicted) != len(sequences) or not set(predicted).issubset(set(LABELS)):
            raise AssertionError(f"{type(model).__name__} returned invalid labels.")
    if dag.root not in {("F", "I"), ("F", "B"), ("I", "B")}:
        raise AssertionError(f"DAG root {dag.root} is not a pairwise node.")
    print("Smoke test passed.")


def _percent(metrics: dict[str, object], key: str) -> float:
    block = metrics[key]
    assert isinstance(block, dict)
    return 100.0 * float(block["mean"])


def paper_gap(metrics: dict[str, object]) -> float:
    """Sum of absolute gaps, in percentage points, from the three published accuracies."""

    return sum(abs(_percent(metrics, key) - target) for key, target in PAPER_ACCURACY.items())


def handle_cut_features(
    recordings: list[Recording],
    cache_path: Path,
) -> list[NDArray[np.float64]]:
    """Features after removing the published 0.5 s handles, without the statistical VAD."""

    cached = load_feature_cache(cache_path, recordings, "handle-cut-0.5")
    if cached is not None:
        print(f"Loaded handle-cut features from {cache_path}")
        return cached
    extractor = FeatureExtractor()
    sequences: list[NDArray[np.float64]] = []
    cut = int(0.5 * 8000)
    for recording in tqdm(recordings, desc="handle-cut"):
        audio = load_audio(recording.path)
        if len(audio) > 2 * cut + 240:
            audio = np.ascontiguousarray(audio[cut:-cut])
        sequences.append(extractor.transform(audio))
    save_feature_cache(cache_path, sequences, recordings, "handle-cut-0.5")
    print(f"Saved handle-cut features to {cache_path}")
    return sequences


def run_search(
    sequences: list[NDArray[np.floating]],
    recordings: list[Recording],
    search_dir: Path,
    n_jobs: int,
    seed: int,
    handle_sequences: list[NDArray[np.floating]] | None = None,
) -> dict[str, object]:
    """Screen feature and HMM choices, then refit the closest setting on the full grid."""

    search_dir = Path(search_dir)
    search_dir.mkdir(parents=True, exist_ok=True)
    experiments: list[tuple[str, list[NDArray[np.floating]], str, str, str, bool]] = [
        ("both-ergodic-fold-raw", list(sequences), "both", "fold", "ergodic", False),
        ("mfcc-ergodic-fold-raw", list(sequences), "mfcc", "fold", "ergodic", False),
        ("tm-ergodic-fold-raw", list(sequences), "tm", "fold", "ergodic", False),
        ("both-ltr-fold-raw", list(sequences), "both", "fold", "left-to-right", False),
        ("both-ergodic-none-raw", list(sequences), "both", "none", "ergodic", False),
        ("both-ergodic-utterance-raw", list(sequences), "both", "utterance", "ergodic", False),
        ("both-ergodic-fold-lennorm", list(sequences), "both", "fold", "ergodic", True),
    ]
    if handle_sequences is not None:
        experiments.append(
            ("mfcc-handles-ergodic-fold-raw", list(handle_sequences), "mfcc", "fold", "ergodic", False)
        )
        experiments.append(
            ("both-handles-ergodic-fold-raw", list(handle_sequences), "both", "fold", "ergodic", False)
        )

    rows: list[dict[str, object]] = []
    for name, data, view, scaler, topology, length_normalize in experiments:
        print(f"\n=== screen {name} ===", flush=True)
        config = ModelConfig(
            state_grid=(5,),
            mix_grid=(2,),
            n_iter=10,
            n_jobs=n_jobs,
            seed=seed,
            topology=topology,
            length_normalize=length_normalize,
        )
        metrics = evaluate(
            data,
            recordings,
            search_dir / name,
            model_config=config,
            feature_view=view,
            scaler_mode=scaler,
        )
        row = {
            "name": name,
            "feature_view": view,
            "scaler_mode": scaler,
            "topology": topology,
            "length_normalize": length_normalize,
            "source": "handle-cut" if "handles" in name else "vad",
            "dag_hmm": _percent(metrics, "dag_hmm"),
            "class_specific_hmm": _percent(metrics, "class_specific_hmm"),
            "rbf_svm": _percent(metrics, "rbf_svm"),
            "recall": metrics["recall"],
            "gap": paper_gap(metrics),
        }
        rows.append(row)
        print(
            f"{name}: DAG {row['dag_hmm']:.2f}  HMM {row['class_specific_hmm']:.2f}  "
            f"SVM {row['rbf_svm']:.2f}  gap {row['gap']:.2f}",
            flush=True,
        )

    best = min(rows, key=lambda item: (float(item["gap"]), -float(item["dag_hmm"])))
    print(f"\nClosest screen: {best['name']} (gap {float(best['gap']):.2f})", flush=True)
    full_config = ModelConfig(
        n_jobs=n_jobs,
        seed=seed,
        topology=str(best["topology"]),
        length_normalize=bool(best["length_normalize"]),
    )
    full_sequences = handle_sequences if best["source"] == "handle-cut" else sequences
    full_metrics = evaluate(
        list(full_sequences) if full_sequences is not None else list(sequences),
        recordings,
        search_dir / "best-full",
        model_config=full_config,
        feature_view=str(best["feature_view"]),
        scaler_mode=str(best["scaler_mode"]),
    )
    summary = {
        "screen": rows,
        "best_screen": best,
        "full_grid": {
            "dag_hmm": _percent(full_metrics, "dag_hmm"),
            "class_specific_hmm": _percent(full_metrics, "class_specific_hmm"),
            "rbf_svm": _percent(full_metrics, "rbf_svm"),
            "recall": full_metrics["recall"],
            "gap": paper_gap(full_metrics),
            "feature_view": best["feature_view"],
            "scaler_mode": best["scaler_mode"],
            "topology": best["topology"],
            "length_normalize": best["length_normalize"],
            "source": best["source"],
        },
        "paper_accuracy": PAPER_ACCURACY,
        "paper_recall": PAPER_RECALL,
    }
    (search_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFull grid on the closest setting:", flush=True)
    _print_summary(full_metrics)
    return summary


def _class_counts(labels: NDArray[np.str_]) -> dict[str, int]:
    return {label: int(np.sum(labels == label)) for label in LABELS}


def _majority_label(counts: dict[str, int]) -> str:
    """Most common training label. Ties break in F, I, B order."""

    return max(LABELS, key=lambda label: (counts[label], -LABELS.index(label)))


def _fixed_states(counts: dict[str, int]) -> int:
    """Use 3 states unless a class is too small for the inner holdout."""

    smallest = min(counts[label] for label in LABELS)
    trainable = smallest if smallest <= 1 else smallest - 1
    return max(1, min(3, trainable))


def _recall_percent(
    predicted: NDArray[np.str_],
    truth: NDArray[np.str_],
) -> dict[str, float | None]:
    recall: dict[str, float | None] = {}
    for label in LABELS:
        mask = truth == label
        if not np.any(mask):
            recall[label] = None
            continue
        recall[label] = 100.0 * float(np.mean(predicted[mask] == truth[mask]))
    return recall


def _score_predictions(
    predicted: NDArray[np.str_],
    truth: NDArray[np.str_],
) -> dict[str, object]:
    return {
        "accuracy": _accuracy(predicted, truth),
        "recall": _recall_percent(predicted, truth),
    }


def _mean_std(values: list[float]) -> dict[str, float | list[float] | None]:
    if not values:
        return {"mean": None, "std": None, "values": []}
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "mean": float(np.mean(array)),
        "std": std,
        "values": [float(value) for value in values],
    }


def _direction_summary(rows: list[dict[str, object]], pool_key: str) -> dict[str, object]:
    """Unweighted mean across training cats for one test pool."""

    def pool_of(row: dict[str, object]) -> dict[str, object]:
        pool = row[pool_key]
        assert isinstance(pool, dict)
        return pool

    summary: dict[str, object] = {
        "n_trainers": len(rows),
        "majority": _mean_std(
            [float(pool_of(row)["majority_accuracy"]) for row in rows]
        ),
    }
    for model_name in ("dag_hmm", "class_specific_hmm", "rbf_svm"):
        accuracies: list[float] = []
        recall_values: dict[str, list[float]] = {label: [] for label in LABELS}
        for row in rows:
            block = pool_of(row)[model_name]
            assert isinstance(block, dict)
            accuracies.append(float(block["accuracy"]))
            recall = block["recall"]
            assert isinstance(recall, dict)
            for label in LABELS:
                if recall[label] is not None:
                    recall_values[label].append(float(recall[label]))
        summary[model_name] = _mean_std(accuracies)
        summary[f"{model_name}_recall"] = {
            label: _mean_std(recall_values[label]) for label in LABELS
        }
    return summary


def run_one_cat(
    sequences: list[NDArray[np.floating]],
    recordings: list[Recording],
    results_dir: Path,
    n_jobs: int,
    seed: int,
) -> dict[str, object]:
    """Train each three-context cat alone and score same-breed and other-breed cats."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([recording.label for recording in recordings], dtype=str)
    cat_ids = np.asarray([recording.cat_id for recording in recordings], dtype=str)
    breeds = np.asarray([recording.breed for recording in recordings], dtype=str)
    by_cat: dict[str, NDArray[np.int_]] = {
        cat_id: np.flatnonzero(cat_ids == cat_id).astype(int)
        for cat_id in sorted(set(cat_ids.tolist()))
    }

    skipped: list[dict[str, object]] = []
    trainers: list[tuple[str, str, NDArray[np.int_], dict[str, int]]] = []
    for cat_id, indices in by_cat.items():
        counts = _class_counts(labels[indices])
        breed = str(breeds[int(indices[0])])
        if min(counts.values()) < 1:
            missing = [label for label in LABELS if counts[label] == 0]
            skipped.append(
                {
                    "cat_id": cat_id,
                    "breed": breed,
                    "n": int(len(indices)),
                    "class_counts": counts,
                    "reason": f"missing {', '.join(missing)}",
                }
            )
            continue
        trainers.append((cat_id, breed, indices, counts))

    print(
        f"One-cat transfer: {len(trainers)} trainers, {len(skipped)} cats skipped.",
        flush=True,
    )
    for row in skipped:
        print(
            f"  skip {row['cat_id']} {row['breed']}: {row['reason']} {row['class_counts']}",
            flush=True,
        )

    fitted: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for cat_id, breed, train_idx, counts in trainers:
        majority = _majority_label(counts)
        state_attempts = [_fixed_states(counts)]
        if state_attempts[0] != 1:
            state_attempts.append(1)
        train_sequences = [sequences[int(index)] for index in train_idx]
        same_idx = np.flatnonzero((breeds == breed) & (cat_ids != cat_id)).astype(int)
        other_idx = np.flatnonzero(breeds != breed).astype(int)
        same_sequences = [sequences[int(index)] for index in same_idx]
        other_sequences = [sequences[int(index)] for index in other_idx]
        train_scaled, same_scaled = scale_sequences(train_sequences, same_sequences)
        _, other_scaled = scale_sequences(train_sequences, other_sequences)
        print(
            f"\n{cat_id} {breed} n={len(train_idx)} "
            f"counts={counts} majority={majority}",
            flush=True,
        )
        dag: DAGHMM | None = None
        class_hmm: ClassSpecificHMM | None = None
        svm: UtteranceRbfSVM | None = None
        n_states = state_attempts[0]
        last_error = "fit was not attempted"
        for n_states in state_attempts:
            config = ModelConfig(
                state_grid=(n_states,),
                mix_grid=(1,),
                n_iter=25,
                n_jobs=n_jobs,
                seed=seed,
            )
            try:
                dag = DAGHMM(config).fit(train_scaled, labels[train_idx])
                class_hmm = ClassSpecificHMM(config).fit(train_scaled, labels[train_idx])
                svm = UtteranceRbfSVM(config).fit(train_scaled, labels[train_idx])
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                dag = None
                print(f"  states={n_states} failed: {last_error}", flush=True)
        if dag is None or class_hmm is None or svm is None:
            failures.append(
                {
                    "cat_id": cat_id,
                    "breed": breed,
                    "n": int(len(train_idx)),
                    "class_counts": counts,
                    "n_states": n_states,
                    "error": last_error,
                }
            )
            continue
        print(f"  fitted with {n_states} states, 1 mixture", flush=True)

        pools: dict[str, object] = {}
        for pool_name, pool_idx, pool_sequences in (
            ("same_breed", same_idx, same_scaled),
            ("other_breed", other_idx, other_scaled),
        ):
            truth = labels[pool_idx]
            pool = {
                "n": int(len(pool_idx)),
                "class_counts": _class_counts(truth),
                "majority_label": majority,
                "majority_accuracy": float(np.mean(truth == majority)),
                "dag_hmm": _score_predictions(dag.predict(pool_sequences), truth),
                "class_specific_hmm": _score_predictions(class_hmm.predict(pool_sequences), truth),
                "rbf_svm": _score_predictions(svm.predict(pool_sequences), truth),
            }
            pools[pool_name] = pool
            print(
                f"  {pool_name}: n={pool['n']}  "
                f"DAG {100.0 * float(pool['dag_hmm']['accuracy']):.1f}%  "
                f"HMM {100.0 * float(pool['class_specific_hmm']['accuracy']):.1f}%  "
                f"SVM {100.0 * float(pool['rbf_svm']['accuracy']):.1f}%  "
                f"majority {100.0 * float(pool['majority_accuracy']):.1f}%",
                flush=True,
            )
        fitted.append(
            {
                "cat_id": cat_id,
                "breed": breed,
                "n": int(len(train_idx)),
                "class_counts": counts,
                "n_states": n_states,
                "n_mix": 1,
                "majority_label": majority,
                "same_breed": pools["same_breed"],
                "other_breed": pools["other_breed"],
            }
        )

    directions = {
        "MC_to_MC": _direction_summary(
            [row for row in fitted if row["breed"] == "MC"], "same_breed"
        ),
        "MC_to_EU": _direction_summary(
            [row for row in fitted if row["breed"] == "MC"], "other_breed"
        ),
        "EU_to_EU": _direction_summary(
            [row for row in fitted if row["breed"] == "EU"], "same_breed"
        ),
        "EU_to_MC": _direction_summary(
            [row for row in fitted if row["breed"] == "EU"], "other_breed"
        ),
    }
    summary = {
        "hmm": {
            "states": "min(3, smallest class count minus one holdout clip)",
            "mixtures": 1,
            "n_iter": 25,
            "topology": "ergodic",
            "length_normalize": False,
            "features": "both",
            "scaler": "training cat frames only",
        },
        "skipped": skipped,
        "failures": failures,
        "trainers": fitted,
        "directions": directions,
    }
    out_path = results_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", flush=True)
    _print_directions(directions)
    return summary


def _print_directions(directions: dict[str, object]) -> None:
    print("\nMean accuracy by transfer direction:")
    for name, block in directions.items():
        assert isinstance(block, dict)
        parts = [f"{name} (n={block['n_trainers']})"]
        for key, title in (
            ("dag_hmm", "DAG"),
            ("class_specific_hmm", "HMM"),
            ("rbf_svm", "SVM"),
            ("majority", "majority"),
        ):
            stats = block[key]
            assert isinstance(stats, dict)
            mean = stats["mean"]
            text = "n/a" if mean is None else f"{100.0 * float(mean):.1f}%"
            parts.append(f"{title} {text}")
        print("  " + "  ".join(parts), flush=True)
