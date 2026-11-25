"""Command-line entry point for the CatMeows DAG-HMM replication.

Environment variables that control BLAS threads are set before NumPy is
imported. Each GMM-HMM fit is already a small Python job, so one BLAS thread
per process keeps the Ryzen 7600X from oversubscribing its 6 cores.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

from src.dataset import (
    SEED,
    VadConfig,
    discover,
    download_catmeows,
    missing_data_message,
)
from src.evaluate import (
    evaluate,
    handle_cut_features,
    prepare_features,
    run_one_cat,
    run_search,
    run_smoke,
)
from src.features import FeatureConfig
from src.models import ModelConfig, quick_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate Ntalampiras et al. (2019) cat-vocalization classification "
            "on the public CatMeows corpus."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory of CatMeows wav files (default: data/raw).",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/features_cache.npz"),
        help="Feature cache path (default: data/features_cache.npz).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for metrics.json and the confusion matrix.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset.zip from Zenodo into --data-dir before training.",
    )
    parser.add_argument(
        "--group-by-cat",
        action="store_true",
        help="Evaluate with GroupKFold on cat id instead of stratified recording folds.",
    )
    parser.add_argument(
        "--one-cat",
        action="store_true",
        help=(
            "Train each cat that has all three contexts by itself, then score "
            "same-breed and other-breed cats."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced state/mixture grid and 10 EM iterations.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run synthetic checks and exit without the corpus.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "Screen feature streams, HMM topology, scaling, and length-normalized "
            "likelihoods, then refit the closest setting on the full grid."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=6,
        help="Parallel GMM-HMM fits. Use 6 on a Ryzen 7600X (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for the outer split and model initialization.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_jobs < 1:
        print("--n-jobs must be at least 1.", file=sys.stderr)
        return 2
    if args.smoke:
        run_smoke()
        return 0

    if args.download:
        download_catmeows(args.data_dir)

    recordings, skipped = discover(args.data_dir)
    if skipped:
        print(f"Skipped {len(skipped)} wav files that do not match the CatMeows pattern.")
    if not recordings:
        print(missing_data_message(args.data_dir), file=sys.stderr)
        return 1

    counts = {label: 0 for label in ("F", "I", "B")}
    for recording in recordings:
        counts[recording.label] = counts.get(recording.label, 0) + 1
    print(
        f"Found {len(recordings)} recordings "
        f"(F={counts.get('F', 0)}, I={counts.get('I', 0)}, B={counts.get('B', 0)})."
    )

    sequences = prepare_features(
        recordings,
        args.cache,
        feature_config=FeatureConfig(),
        vad_config=VadConfig(),
    )
    if args.one_cat:
        run_one_cat(
            sequences,
            recordings,
            results_dir=args.results_dir / "one-cat",
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        return 0

    if args.search:
        handle_sequences = handle_cut_features(
            recordings, args.cache.with_name("features_cache_handles.npz")
        )
        run_search(
            sequences,
            recordings,
            search_dir=args.results_dir / "search",
            n_jobs=args.n_jobs,
            seed=args.seed,
            handle_sequences=handle_sequences,
        )
        return 0

    if args.quick:
        model_config = quick_model_config(n_jobs=args.n_jobs, seed=args.seed)
    else:
        model_config = ModelConfig(n_jobs=args.n_jobs, seed=args.seed)
    evaluate(
        sequences,
        recordings,
        results_dir=args.results_dir,
        group_by_cat=args.group_by_cat,
        model_config=model_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
