"""Shared feature-extraction and scoring implementation for the sample tasks."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GENERATORS = ("GPT", "Gemini", "Claude")
LENGTHS = ("short", "medium", "long")
SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
N_BOOTSTRAP = 1000
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
BERTSCORE_MODEL = "roberta-large"
BATCH_SIZE = 64

FEATURE_COLUMNS = [
    "similarity_mean",
    "similarity_max",
    "bert_P_mean",
    "bert_R_mean",
    "bert_F1_mean",
    "bert_F1_max",
]
FEATURE_SETS = {
    "cosine": ["similarity_mean"],
    "bertscore": ["bert_P_mean", "bert_R_mean", "bert_F1_mean"],
    "combined": ["similarity_mean", "bert_P_mean", "bert_R_mean", "bert_F1_mean"],
}


@dataclass(frozen=True)
class PipelineConfig:
    task: str
    score_min: int
    score_max: int
    dataset_csv: Path
    references_csv: Path
    output_dir: Path
    id_column: str = "ID"
    text_column: str = "essay"
    score_column: str = "score"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def load_dataset(config: PipelineConfig) -> pd.DataFrame:
    frame = read_csv_flexible(config.dataset_csv)
    required = {config.id_column, config.text_column, config.score_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{config.dataset_csv} is missing columns: {sorted(missing)}")
    frame = (
        frame[[config.id_column, config.text_column, config.score_column]]
        .rename(
            columns={
                config.id_column: "ID",
                config.text_column: "essay",
                config.score_column: "score",
            }
        )
        .dropna(subset=["score"])
        .drop_duplicates("ID")
        .sort_values("ID")
        .reset_index(drop=True)
    )
    frame["essay"] = frame["essay"].fillna("").astype(str)
    frame["score"] = pd.to_numeric(frame["score"], errors="raise").astype(int)
    if frame.empty:
        raise ValueError("The dataset contains no usable responses.")
    if frame.score.min() < config.score_min or frame.score.max() > config.score_max:
        raise ValueError(
            f"Observed scores {frame.score.min()}--{frame.score.max()} fall outside "
            f"the configured range {config.score_min}--{config.score_max}."
        )
    return frame


def load_references(config: PipelineConfig) -> dict[tuple[str, str], list[str]]:
    frame = read_csv_flexible(config.references_csv)
    required = {"task", "generator", "condition", "reference_answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{config.references_csv} is missing columns: {sorted(missing)}")
    frame = frame[frame.task.astype(str).str.casefold() == config.task.casefold()].copy()
    references: dict[tuple[str, str], list[str]] = {}
    for generator in GENERATORS:
        for length in LENGTHS:
            selected = frame[
                frame.generator.astype(str).str.casefold().eq(generator.casefold())
                & frame.condition.astype(str).str.casefold().eq(length)
            ]
            texts = [
                text.strip()
                for text in selected.reference_answer.dropna().astype(str)
                if text.strip()
            ]
            if len(texts) != 5:
                raise ValueError(
                    f"{config.task}/{generator}/{length}: expected 5 references, "
                    f"found {len(texts)}"
                )
            references[(generator, length)] = texts
    signatures = [tuple(texts) for texts in references.values()]
    if len(set(signatures)) != len(signatures):
        raise ValueError("Identical reference sets were found in different conditions.")
    return references


def extract_features(
    frame: pd.DataFrame,
    references: dict[tuple[str, str], list[str]],
    config: PipelineConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    import torch
    from bert_score import score as bertscore
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    essays = frame.essay.tolist()
    model = SentenceTransformer(SBERT_MODEL, device=device)
    start = time.time()
    essay_vectors = model.encode(
        essays,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Encoded {len(essays)} responses in {time.time() - start:.0f} seconds on {device}.")

    feature_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for (generator, length), texts in references.items():
        reference_vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        similarities = essay_vectors @ reference_vectors.T
        candidates = [essay for essay in essays for _ in texts]
        repeated_references = [text for _ in essays for text in texts]
        precision, recall, f1 = bertscore(
            candidates,
            repeated_references,
            model_type=BERTSCORE_MODEL,
            lang="en",
            batch_size=BATCH_SIZE,
            device=device,
            rescale_with_baseline=False,
            verbose=False,
        )
        n_responses, n_references = len(frame), len(texts)
        precision = precision.cpu().numpy().reshape(n_responses, n_references)
        recall = recall.cpu().numpy().reshape(n_responses, n_references)
        f1 = f1.cpu().numpy().reshape(n_responses, n_references)
        features = frame.copy()
        features["similarity_mean"] = similarities.mean(axis=1)
        features["similarity_max"] = similarities.max(axis=1)
        features["bert_P_mean"] = precision.mean(axis=1)
        features["bert_R_mean"] = recall.mean(axis=1)
        features["bert_F1_mean"] = f1.mean(axis=1)
        features["bert_F1_max"] = f1.max(axis=1)
        features["task"] = config.task
        features["generator"] = generator
        features["length"] = length
        if features[FEATURE_COLUMNS].isna().any().any():
            raise RuntimeError(f"Missing feature values for {generator}/{length}.")
        feature_frames[(generator, length)] = features
    return feature_frames


def clip_round(values: np.ndarray, score_min: int, score_max: int) -> np.ndarray:
    return np.clip(np.rint(values), score_min, score_max).astype(int)


def run_condition(
    data: pd.DataFrame,
    features: list[str],
    seed: int,
    config: PipelineConfig,
) -> dict[str, object]:
    train, remainder = train_test_split(
        data, test_size=0.4, random_state=seed, shuffle=True
    )
    validation, test = train_test_split(
        remainder, test_size=0.5, random_state=seed, shuffle=True
    )
    x_train, y_train = train[features].values, train.score.values
    x_validation, y_validation = validation[features].values, validation.score.values
    x_test, y_test = test[features].values, test.score.values

    def qwk_score(y_true: np.ndarray, prediction: np.ndarray) -> float:
        rounded = clip_round(prediction, config.score_min, config.score_max)
        return cohen_kappa_score(y_true.astype(int), rounded, weights="quadratic")

    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": ALPHAS},
        cv=KFold(5, shuffle=True, random_state=seed),
        scoring=make_scorer(qwk_score),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    alpha = float(search.best_params_["ridge__alpha"])

    x_train_validation = np.vstack([x_train, x_validation])
    y_train_validation = np.concatenate([y_train, y_validation])
    final_model = Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )
    final_model.fit(x_train_validation, y_train_validation)
    continuous = final_model.predict(x_test)
    rounded = clip_round(continuous, config.score_min, config.score_max)
    return {
        "alpha": alpha,
        "qwk": cohen_kappa_score(y_test, rounded, weights="quadratic"),
        "r": pearsonr(y_test, continuous)[0],
        "mae": mean_absolute_error(y_test, continuous),
        "prediction_continuous": continuous,
        "prediction_rounded": rounded,
        "ids": test.ID.to_numpy(),
        "true_score": y_test,
    }


def paired_bootstrap(
    first: dict[str, object],
    second: dict[str, object],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    first_ids = np.asarray(first["ids"])
    second_ids = np.asarray(second["ids"])
    first_scores = np.asarray(first["true_score"])
    second_scores = np.asarray(second["true_score"])
    if not np.array_equal(first_ids, second_ids) or not np.array_equal(
        first_scores, second_scores
    ):
        raise ValueError("Paired conditions do not contain identical held-out responses.")
    first_predictions = np.asarray(first["prediction_rounded"])
    second_predictions = np.asarray(second["prediction_rounded"])
    differences = np.empty(N_BOOTSTRAP)
    for index in range(N_BOOTSTRAP):
        sample = rng.integers(0, len(first_scores), len(first_scores))
        differences[index] = cohen_kappa_score(
            first_scores[sample], first_predictions[sample], weights="quadratic"
        ) - cohen_kappa_score(
            first_scores[sample], second_predictions[sample], weights="quadratic"
        )
    return (
        float(np.nanmean(differences)),
        float(np.nanpercentile(differences, 2.5)),
        float(np.nanpercentile(differences, 97.5)),
    )


def summarize_qwk(results: pd.DataFrame, task: str) -> pd.DataFrame:
    grouped = results.groupby(["generator", "length", "feature_set"])["test_qwk"]
    summary = grouped.agg(mean="mean", sd="std", n="count").reset_index()
    half_width = 1.96 * summary.sd / np.sqrt(summary.n)
    summary["ci_lo"] = summary["mean"] - half_width
    summary["ci_hi"] = summary["mean"] + half_width
    summary.insert(0, "task", task)
    return summary.drop(columns="n")


def comparison_row(
    task: str,
    test: str,
    condition: str,
    first: str,
    second: str,
    estimates: list[tuple[float, float, float]],
) -> dict[str, object]:
    significant = sum(low > 0 or high < 0 for _, low, high in estimates)
    return {
        "task": task,
        "test": test,
        "condition": condition,
        "pair": f"{first}-{second}",
        "mean_diff": np.nanmean([value for value, _, _ in estimates]),
        "ci_lo": np.nanmean([low for _, low, _ in estimates]),
        "ci_hi": np.nanmean([high for _, _, high in estimates]),
        "seeds_significant": f"{significant}/{len(SEEDS)}",
    }


def run_pipeline(config: PipelineConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    dataset = load_dataset(config)
    references = load_references(config)
    feature_frames = extract_features(dataset, references, config)

    feature_dir = config.output_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    for (generator, length), frame in feature_frames.items():
        public_columns = ["ID", "score", *FEATURE_COLUMNS, "task", "generator", "length"]
        frame[public_columns].to_csv(
            feature_dir / f"features_{config.task}_{generator}_{length}.csv", index=False
        )

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    stored: dict[tuple[str, str, str, int], dict[str, object]] = {}
    for (generator, length), frame in feature_frames.items():
        for feature_set, columns in FEATURE_SETS.items():
            for seed in SEEDS:
                result = run_condition(frame, columns, seed, config)
                metric_rows.append(
                    {
                        "task": config.task,
                        "generator": generator,
                        "length": length,
                        "feature_set": feature_set,
                        "seed": seed,
                        "alpha": result["alpha"],
                        "test_qwk": result["qwk"],
                        "test_r": result["r"],
                        "test_mae": result["mae"],
                    }
                )
                prediction_rows.extend(
                    {
                        "task": config.task,
                        "generator": generator,
                        "length": length,
                        "feature_set": feature_set,
                        "seed": seed,
                        "ID": response_id,
                        "true_score": true_score,
                        "prediction_continuous": continuous,
                        "prediction_rounded": rounded,
                        "alpha": result["alpha"],
                    }
                    for response_id, true_score, continuous, rounded in zip(
                        result["ids"],
                        result["true_score"],
                        result["prediction_continuous"],
                        result["prediction_rounded"],
                    )
                )
                stored[(generator, length, feature_set, seed)] = result

    results = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    results["run_timestamp"] = run_stamp
    predictions["run_timestamp"] = run_stamp
    results.to_csv(config.output_dir / f"{config.task}_raw_{run_stamp}.csv", index=False)
    predictions.to_csv(
        config.output_dir / f"{config.task}_predictions_{run_stamp}.csv", index=False
    )
    summary = summarize_qwk(results, config.task).round(4)
    summary["run_timestamp"] = run_stamp
    summary.to_csv(
        config.output_dir / f"{config.task}_summary_{run_stamp}.csv", index=False
    )

    rng = np.random.default_rng(0)
    significance_rows: list[dict[str, object]] = []
    for generator in GENERATORS:
        for first_length, second_length in (
            ("medium", "short"),
            ("long", "short"),
            ("long", "medium"),
        ):
            estimates = [
                paired_bootstrap(
                    stored[(generator, first_length, "combined", seed)],
                    stored[(generator, second_length, "combined", seed)],
                    rng,
                )
                for seed in SEEDS
            ]
            significance_rows.append(
                comparison_row(
                    config.task,
                    "length",
                    generator,
                    first_length,
                    second_length,
                    estimates,
                )
            )
    for length in LENGTHS:
        for first_generator, second_generator in itertools.combinations(GENERATORS, 2):
            estimates = [
                paired_bootstrap(
                    stored[(first_generator, length, "combined", seed)],
                    stored[(second_generator, length, "combined", seed)],
                    rng,
                )
                for seed in SEEDS
            ]
            significance_rows.append(
                comparison_row(
                    config.task,
                    "generator",
                    length,
                    first_generator,
                    second_generator,
                    estimates,
                )
            )
    significance = pd.DataFrame(significance_rows)
    significance["run_timestamp"] = run_stamp
    significance.to_csv(
        config.output_dir / f"{config.task}_significance_{run_stamp}.csv", index=False
    )

    ablation_rows: list[dict[str, object]] = []
    for generator in GENERATORS:
        for first_set, second_set in (
            ("bertscore", "cosine"),
            ("combined", "bertscore"),
            ("combined", "cosine"),
        ):
            estimates = [
                paired_bootstrap(
                    stored[(generator, "long", first_set, seed)],
                    stored[(generator, "long", second_set, seed)],
                    rng,
                )
                for seed in SEEDS
            ]
            ablation_rows.append(
                comparison_row(
                    config.task,
                    "feature_set",
                    generator,
                    first_set,
                    second_set,
                    estimates,
                )
            )
    ablation = pd.DataFrame(ablation_rows)
    ablation["length"] = "long"
    ablation["run_timestamp"] = run_stamp
    ablation.to_csv(
        config.output_dir / f"{config.task}_ablation_{run_stamp}.csv", index=False
    )
    print(f"Outputs saved to {config.output_dir}")
