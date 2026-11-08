"""Measure semantic and lexical diversity among generated references.

The generated-reference analysis runs without the datasets. If independently
downloaded task CSVs are placed in ``data/`` (or ``AES_DATA_DIR``), the script
also adds the optional high-scoring-student comparison reported in the audit.
"""

from __future__ import annotations

import itertools
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REFERENCES_CSV = Path(
    os.environ.get(
        "AES_REFERENCES_CSV",
        PACKAGE_ROOT
        / "artifacts"
        / "generated_references"
        / "all_generated_references.csv",
    )
)
DATA_DIR = Path(os.environ.get("AES_DATA_DIR", PACKAGE_ROOT / "data"))
OUTPUT_DIR = Path(
    os.environ.get(
        "AES_DIVERSITY_OUTPUT_DIR",
        PACKAGE_ROOT / "local_outputs" / "reference_diversity",
    )
)
SEED = 42
N_STUDENT_PAIRS = 200
TASKS = tuple([f"ASAP{i}" for i in range(1, 9)] + [f"OS{i}" for i in range(1, 6)])


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
            frame.columns = [str(c).replace("\ufeff", "").strip() for c in frame.columns]
            return frame
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def find_dataset(task: str) -> Path | None:
    number = task.removeprefix("ASAP").removeprefix("OS")
    candidates = (
        DATA_DIR / f"{task}.csv",
        DATA_DIR / f"{task.lower()}.csv",
        DATA_DIR / f"ASAP_task{number}.csv",
        DATA_DIR / f"OS_task{number}.csv",
    )
    return next((path for path in candidates if path.exists()), None)


def load_references() -> pd.DataFrame:
    frame = read_csv_flexible(REFERENCES_CSV)
    required = {"task", "generator", "condition", "reference_answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{REFERENCES_CSV} is missing columns: {sorted(missing)}")
    frame = frame[list(required)].rename(columns={"reference_answer": "text"})
    frame["text"] = frame.text.fillna("").astype(str).str.strip()
    frame = frame[frame.text.ne("")].copy()
    counts = frame.groupby(["task", "generator", "condition"]).size()
    bad = counts[counts.ne(5)]
    if len(bad):
        raise ValueError(f"Expected five references in every set:\n{bad}")
    frame["ref_id"] = frame.groupby(["task", "generator", "condition"]).cumcount() + 1
    return frame


def load_optional_students() -> dict[str, pd.DataFrame]:
    students: dict[str, pd.DataFrame] = {}
    for task in TASKS:
        path = find_dataset(task)
        if path is None:
            continue
        frame = read_csv_flexible(path)
        required = {"essay", "score"}
        if required.difference(frame.columns):
            print(f"Skipping student comparison for {task}: missing essay/score columns.")
            continue
        selected = frame[["essay", "score"]].dropna().rename(columns={"essay": "text"})
        selected["score"] = pd.to_numeric(selected.score, errors="raise")
        students[task] = selected
    return students


def build_pairs(refs: pd.DataFrame, students: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pairs: list[tuple[str, str, str, str, str, str]] = []
    for (task, condition), group in refs.groupby(["task", "condition"]):
        by_generator = {
            generator: subset.text.tolist()
            for generator, subset in group.groupby("generator")
        }
        for generator, texts in by_generator.items():
            for first, second in itertools.combinations(texts, 2):
                pairs.append(("within_set", task, condition, generator, first, second))
        for first_generator, second_generator in itertools.combinations(
            sorted(by_generator), 2
        ):
            for first in by_generator[first_generator]:
                for second in by_generator[second_generator]:
                    pairs.append(
                        (
                            "cross_generator",
                            task,
                            condition,
                            f"{first_generator}-{second_generator}",
                            first,
                            second,
                        )
                    )

    rng = random.Random(SEED)
    for task, frame in students.items():
        cutoff = frame.score.quantile(0.90)
        texts = frame.loc[frame.score >= cutoff, "text"].astype(str).tolist()
        indices = list(itertools.combinations(range(len(texts)), 2))
        rng.shuffle(indices)
        for first, second in indices[:N_STUDENT_PAIRS]:
            pairs.append(
                (
                    "student_top",
                    task,
                    "all",
                    f"score>={cutoff:g} (n={len(texts)})",
                    texts[first],
                    texts[second],
                )
            )
    return pd.DataFrame(
        pairs,
        columns=["comparison", "task", "condition", "label", "text_a", "text_b"],
    )


def ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def add_metrics(pairs: pd.DataFrame) -> pd.DataFrame:
    import torch
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert = SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2", device=device
    )
    unique_texts = pd.unique(pd.concat([pairs.text_a, pairs.text_b], ignore_index=True))
    embeddings = sbert.encode(
        unique_texts.tolist(),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    lookup = {text: index for index, text in enumerate(unique_texts)}
    first = embeddings[pairs.text_a.map(lookup).to_numpy()]
    second = embeddings[pairs.text_b.map(lookup).to_numpy()]
    pairs["sbert_cos"] = (first * second).sum(axis=1)
    del sbert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    scorer = BERTScorer(
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=True,
        device=device,
    )
    _, _, f1 = scorer.score(pairs.text_a.tolist(), pairs.text_b.tolist(), batch_size=32)
    pairs["bertscore_f1_rescaled"] = f1.cpu().numpy()
    baseline_f1 = float(scorer.baseline_vals[2])
    pairs["bertscore_f1_raw"] = (
        pairs.bertscore_f1_rescaled * (1 - baseline_f1) + baseline_f1
    )

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    pairs["rougeL_f1"] = [
        rouge.score(first_text, second_text)["rougeL"].fmeasure
        for first_text, second_text in zip(pairs.text_a, pairs.text_b)
    ]
    pairs["trigram_jaccard"] = [
        len(ngrams(first_text) & ngrams(second_text))
        / max(1, len(ngrams(first_text) | ngrams(second_text)))
        for first_text, second_text in zip(pairs.text_a, pairs.text_b)
    ]
    return pairs


def save_outputs(pairs: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = [
        "sbert_cos",
        "bertscore_f1_rescaled",
        "bertscore_f1_raw",
        "rougeL_f1",
        "trigram_jaccard",
    ]
    pairs["dataset"] = np.where(pairs.task.str.startswith("ASAP"), "ASAP", "OS")
    pairs.drop(columns=["text_a", "text_b"]).to_csv(
        OUTPUT_DIR / "diversity_pairs.csv", index=False
    )

    group_columns = ["comparison", "dataset", "task", "condition", "label"]
    by_set = pairs.groupby(group_columns)[metrics].agg(["mean", "std", "min", "max"])
    by_set.columns = [f"{metric}_{statistic}" for metric, statistic in by_set.columns]
    by_set["n_pairs"] = pairs.groupby(group_columns).size()
    within = pairs[pairs.comparison.eq("within_set")]
    by_set["share_sbert_ge_0.95"] = within.groupby(group_columns).sbert_cos.apply(
        lambda values: values.ge(0.95).mean()
    )
    by_set.reset_index().to_csv(OUTPUT_DIR / "diversity_by_set.csv", index=False)

    pairs["generator"] = np.where(
        pairs.comparison.eq("within_set"), pairs.label, "all"
    )
    summary = pairs.groupby(
        ["comparison", "dataset", "condition", "generator"]
    )[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.round(4).reset_index().to_csv(
        OUTPUT_DIR / "diversity_summary.csv", index=False
    )
    print(f"Outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    references = load_references()
    optional_students = load_optional_students()
    if len(optional_students) != len(TASKS):
        print(
            "Student-response comparison is partial or omitted because datasets are not "
            "included in the package."
        )
    comparison_pairs = build_pairs(references, optional_students)
    save_outputs(add_metrics(comparison_pairs))
