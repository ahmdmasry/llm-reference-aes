# -*- coding: utf-8 -*-
"""
Stratified extreme-case error analysis for the AES manuscript revision (v2).

Reviewer request: "Stratified extreme-case qualitative error analysis using
hybrid deductive-inductive content coding."

Methodological basis
--------------------
Sampling
  * Purposeful sampling (Patton, 2002/2015; Palinkas et al., 2015):
      - extreme (deviant) case sampling: the largest scoring errors;
      - stratified purposeful sampling: a fixed quota per stratum, with strata
        defined by task/question x error direction (over- vs underprediction);

Coding
  * Hybrid deductive-inductive coding (Fereday & Muir-Cochrane, 2006):
      - deductive: an a-priori codebook derived from the scoring rubrics and from
        mechanisms identified in this study (encoder truncation, reference
        coverage, lexical overlap); this corresponds to directed content
        analysis (Hsieh & Shannon, 2005);
      - inductive: codes that emerge from the data are added to the codebook
        with a definition and an example and re-applied to all cases
        (conventional content analysis; Hsieh & Shannon, 2005).
  * Each case receives one primary code and, optionally, one secondary code.

Reliability
  * Two reviewers code independently before consensus. Agreement on the
    primary code is reported as percent agreement and Cohen's kappa
    (Cohen, 1960), computed live in the workbook. Reporting follows
    O'Connor & Joffe (2020); with a small number of cases, kappa is unstable and
    is reported together with percent agreement.

What the script does
--------------------
  1. Reads the saved held-out prediction files of the primary configuration
     (no model is retrained for the primary analysis).
  2. Averages each response over the seeds in which it was held out.
  3. Selects, per task/question, the most extreme overpredicted and
     underpredicted responses.
  4. Adds token counts for both encoders (truncation flags).
  5. For ASAP tasks, retrains the same Ridge protocol on the saved sliding-window
     (full-text) features and reports whether each selected error persists when
     truncation is removed.
  6. Writes a review workbook with codebook, coding columns, reliability
     (percent agreement and kappa) and summary tables driven by formulas.

All outputs go to a new time-stamped folder.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PREDICTION_ROOT = Path(
    os.environ.get(
        "AES_PREDICTION_DIR", PACKAGE_ROOT / "results"
    )
)
DATA_ROOT = Path(os.environ.get("AES_DATA_DIR", PACKAGE_ROOT / "data"))
REFERENCES_CSV = Path(
    os.environ.get(
        "AES_REFERENCES_CSV",
        PACKAGE_ROOT
        / "artifacts"
        / "generated_references"
        / "all_generated_references.csv",
    )
)
SLIDING_RUN = Path(
    os.environ.get(
        "AES_SLIDING_RUN",
        PACKAGE_ROOT / "local_outputs" / "truncation_sensitivity",
    )
)

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(
    os.environ.get(
        "AES_ERROR_OUTPUT_DIR", PACKAGE_ROOT / "local_outputs" / "error_case_analysis"
    )
) / f"run_{RUN_STAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

# ---------------------------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------------------------
TASK_META = {
    "ASAP1": ("ASAP", 1, 2, 12, "Computer's effect on people's lives"),
    "ASAP2": ("ASAP", 2, 2, 10, "Censorship in libraries"),
    "ASAP3": ("ASAP", 3, 0, 3, "Factors affecting the cyclist"),
    "ASAP4": ("ASAP", 4, 0, 3, "Purpose of the concluding paragraph"),
    "ASAP5": ("ASAP", 5, 0, 4, "Mood created in the memoir"),
    "ASAP6": ("ASAP", 6, 0, 4, "Obstacles the builders faced"),
    "ASAP7": ("ASAP", 7, 0, 30, "A story about patience"),
    "ASAP8": ("ASAP", 8, 0, 60, "A story involving laughter"),
    "OS1": ("OS", 1, 0, 5, "What is an operating system?"),
    "OS2": ("OS", 2, 0, 5, "Advantages of a multiprocessor system"),
    "OS3": ("OS", 3, 0, 5, "Main purpose of an operating system"),
    "OS4": ("OS", 4, 0, 3, "How an operating system handles multiple tasks"),
    "OS5": ("OS", 5, 0, 3, "Single-processor versus multiprocessor systems"),
}
TASKS = list(TASK_META)

GENERATOR, LENGTH_CONDITION, FEATURE_SET = "GPT", "long", "combined"

EXTREME_PER_DIRECTION = 1   # per task: this many over- and this many underpredicted cases
MIN_APPEARANCES = 1         # minimum number of seeds in which a response was held out

# Ridge protocol of the paper (Section III-E), used only for the sliding-window check.
FEATURES = ["similarity_mean", "bert_P_mean", "bert_R_mean", "bert_F1_mean"]
SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

SBERT_LIMIT, ROBERTA_LIMIT = 384, 512

# A-priori (deductive) codebook. Reviewers may add inductive codes in the
# empty rows of the Codebook sheet; they appear in the dropdowns automatically.
CODEBOOK = [
    # code, group, definition, basis
    ("Alternative valid expression", "Reference-related",
     "Correct or relevant content is expressed in wording or structure that differs from the references.",
     "Similarity-based scoring mechanism"),
    ("Reference coverage gap", "Reference-related",
     "The response contains a valid concept, interpretation or example that the references do not cover.",
     "Similarity-based scoring mechanism"),
    ("Creative or uncommon response", "Reference-related",
     "A relevant but unusual viewpoint, structure or narrative that is far from all references.",
     "Similarity-based scoring mechanism"),
    ("Keyword-driven overprediction", "Similarity-related",
     "High topical or lexical overlap although the response is incomplete, weak or incorrect.",
     "Similarity-based scoring mechanism"),
    ("Partial or contradictory response", "Content-related",
     "Some relevant content, but key concepts are missing or contradicted.",
     "Scoring rubric (content)"),
    ("Non-semantic writing quality", "Content-related",
     "The human score reflects organization, conventions, coherence, voice or style that semantic similarity does not capture.",
     "Scoring rubric (writing quality)"),
    ("Encoder truncation", "Pipeline-related",
     "The response or references exceed the encoder input limit and content relevant to the score lies beyond it.",
     "Truncation audit and sensitivity analysis (this study)"),
    ("Copied or verbatim source text", "Response-related",
     "The response reproduces the prompt, source text or external material; the human score likely penalizes copying.",
     "Scoring practice"),
    ("Very short, off-topic or non-English response", "Response-related",
     "The response is too short, not on the task, or largely not in English, so similarity is unreliable.",
     "Scoring practice"),
    ("Possible human-scoring anomaly", "Data-related",
     "The human score is hard to reconcile with the response and the rubric.",
     "Data quality"),
]
CODEBOOK_ROWS = 40   # rows reserved in the Codebook sheet (room for inductive codes)


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.columns = [str(c).replace("﻿", "").replace("ï»¿", "").strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def normalized_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if re.fullmatch(r"[-+]?\d+\.0", text) else text


def find_latest_prediction_file(task: str) -> Path:
    pat = re.compile(rf"^{re.escape(task)}_predictions_fold_safe_(\d{{8}}_\d{{4}})\.csv$", re.I)
    found = []
    for p in PREDICTION_ROOT.rglob(f"{task}_predictions_fold_safe_*.csv"):
        if "Error_Case_Analysis" in p.parts:
            continue
        m = pat.match(p.name)
        if m:
            found.append((m.group(1), str(p), p))
    if not found:
        raise FileNotFoundError(f"No prediction file for {task}")
    return sorted(found)[-1][2]


def find_dataset_file(task: str) -> Path:
    ds, num = TASK_META[task][0], TASK_META[task][1]
    name = f"{ds}_task{num}.csv"
    for p in (
        DATA_ROOT / f"{task}.csv",
        DATA_ROOT / name,
        DATA_ROOT / task / name,
    ):
        if p.exists():
            return p
    cands = sorted(DATA_ROOT.rglob(name), key=lambda p: (len(p.parts), str(p)))
    if not cands:
        raise FileNotFoundError(f"Dataset not found for {task}")
    return cands[0]


def load_primary_references(task: str) -> pd.DataFrame:
    frame = read_csv_flexible(REFERENCES_CSV)
    required = {"task", "generator", "condition", "reference_answer"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{REFERENCES_CSV} lacks {sorted(missing)}")
    return frame[
        frame.task.astype(str).str.casefold().eq(task.casefold())
        & frame.generator.astype(str).str.casefold().eq(GENERATOR.casefold())
        & frame.condition.astype(str).str.casefold().eq(LENGTH_CONDITION.casefold())
    ].copy()


# ---------------------------------------------------------------------------
# 4. Load predictions, aggregate per response
# ---------------------------------------------------------------------------
def load_predictions(task: str) -> tuple[pd.DataFrame, Path]:
    path = find_latest_prediction_file(task)
    df = read_csv_flexible(path).rename(columns={"pred_continuous": "prediction_continuous",
                                                 "pred_rounded": "prediction_rounded"})
    need = {"generator", "length", "feature_set", "seed", "ID", "true_score",
            "prediction_continuous", "prediction_rounded"}
    if need - set(df.columns):
        raise ValueError(f"{path.name} lacks {sorted(need - set(df.columns))}")
    sel = df[df.generator.astype(str).str.casefold().eq(GENERATOR.casefold())
             & df.length.astype(str).str.casefold().eq(LENGTH_CONDITION.casefold())
             & df.feature_set.astype(str).str.casefold().eq(FEATURE_SET.casefold())].copy()
    if sel.empty:
        raise RuntimeError(f"No {GENERATOR}/{LENGTH_CONDITION}/{FEATURE_SET} rows in {path}")
    sel["task"] = task
    sel["response_id"] = sel.ID.map(normalized_id)
    for c in ("true_score", "prediction_continuous", "prediction_rounded", "seed"):
        sel[c] = pd.to_numeric(sel[c], errors="raise")
    return sel, path


def aggregate(rows: pd.DataFrame, task: str) -> pd.DataFrame:
    _, _, lo, hi, desc = TASK_META[task]
    rng = hi - lo
    rows = rows.assign(
        signed_error=rows.prediction_rounded - rows.true_score,
        cont_error=rows.prediction_continuous - rows.true_score,
    )
    g = rows.groupby("response_id").agg(
        true_score=("true_score", "first"),
        mean_prediction_continuous=("prediction_continuous", "mean"),
        mean_prediction_rounded=("prediction_rounded", "mean"),
        prediction_sd=("prediction_rounded", "std"),
        mean_signed_error=("signed_error", "mean"),
        mean_absolute_error=("signed_error", lambda s: s.abs().mean()),
        max_absolute_error=("signed_error", lambda s: s.abs().max()),
        mean_abs_continuous_error=("cont_error", lambda s: s.abs().mean()),
        n_exact=("signed_error", lambda s: int((s == 0).sum())),
        test_appearances=("seed", "size"),
        test_seeds=("seed", lambda s: ", ".join(str(int(v)) for v in sorted(set(s)))),
    ).reset_index()
    g["prediction_sd"] = g.prediction_sd.fillna(0.0)
    g["mean_normalized_absolute_error"] = g.mean_absolute_error / rng
    g["mean_normalized_continuous_error"] = g.mean_abs_continuous_error / rng
    g["error_direction"] = np.select([g.mean_signed_error > 0, g.mean_signed_error < 0],
                                     ["Overprediction", "Underprediction"], "Exact or mixed")
    g["task"], g["dataset"], g["task_description"] = task, TASK_META[task][0], desc
    g["score_min"], g["score_max"] = lo, hi
    return g


# ---------------------------------------------------------------------------
# 5. Stratified purposeful selection
# ---------------------------------------------------------------------------
def select_cases(g: pd.DataFrame, texts: pd.DataFrame) -> pd.DataFrame:
    """Select the largest over- and underprediction for each task.

    Cases are ranked by mean normalized rounded-score error; ties are broken by
    continuous-prediction error, held-out appearances, and response ID.
    """
    g = g.merge(texts, on="response_id", how="left", validate="one_to_one")
    if g.student_response.isna().any():
        missing = g.loc[g.student_response.isna(), "response_id"].tolist()[:10]
        raise RuntimeError(f"{g.task.iloc[0]}: response IDs without text, e.g. {missing}")
    pool = g[g.test_appearances >= MIN_APPEARANCES]
    picks = []
    for direction in ("Overprediction", "Underprediction"):
        d = pool[pool.error_direction == direction].sort_values(
            ["mean_normalized_absolute_error", "mean_normalized_continuous_error",
             "test_appearances", "response_id"],
            ascending=[False, False, False, True], kind="mergesort")
        picks.append(d.head(EXTREME_PER_DIRECTION).assign(
            sampling_stratum=f"Extreme: {direction.lower()}",
            stratum_rank=np.arange(1, min(len(d), EXTREME_PER_DIRECTION) + 1)))
    return pd.concat(picks, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Token counts (truncation flags)
# ---------------------------------------------------------------------------
def add_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from transformers import AutoTokenizer
        sb = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        rb = AutoTokenizer.from_pretrained("roberta-large", use_fast=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Token counts skipped ({exc}).")
        df["sbert_tokens"] = df["roberta_tokens"] = np.nan
        df["exceeds_sbert_384"] = df["exceeds_roberta_512"] = ""
        return df
    df["sbert_tokens"] = [len(sb.encode(t, add_special_tokens=True, truncation=False)) for t in df.student_response]
    df["roberta_tokens"] = [len(rb.encode(t, add_special_tokens=True, truncation=False)) for t in df.student_response]
    df["exceeds_sbert_384"] = np.where(df.sbert_tokens > SBERT_LIMIT, "Yes", "No")
    df["exceeds_roberta_512"] = np.where(df.roberta_tokens > ROBERTA_LIMIT, "Yes", "No")
    return df


# ---------------------------------------------------------------------------
# 7. Sliding-window (full-text) check for ASAP tasks
# ---------------------------------------------------------------------------
def sliding_predictions(task: str, original_rows: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Retrain the paper's Ridge protocol on the full-text features and return
    held-out predictions per response, using the same seeds and splits."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import GridSearchCV, KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = f"features_sliding_{LENGTH_CONDITION}_{task}_{GENERATOR}.csv"
    direct = SLIDING_RUN / task / name
    candidates = [direct] if direct.exists() else sorted(SLIDING_RUN.rglob(name))
    path = candidates[-1] if candidates else direct
    if not path.exists():
        return None, f"not available ({path.name} not found)"
    f = read_csv_flexible(path)
    f["response_id"] = f.ID.map(normalized_id)
    f = f.set_index("response_id")
    _, _, lo, hi, _ = TASK_META[task]

    def clip_round(x):
        return np.clip(np.rint(x), lo, hi).astype(int)

    scorer = make_scorer(lambda y, p: cohen_kappa_score(y, clip_round(p), weights="quadratic"))
    ids = np.array(sorted(f.index, key=lambda s: float(s) if re.fullmatch(r"[-+]?\d+(\.\d+)?", s) else s))
    out, mismatched = [], 0
    for seed in SEEDS:
        tr, tmp = train_test_split(ids, test_size=0.4, random_state=seed, shuffle=True)
        va, te = train_test_split(tmp, test_size=0.5, random_state=seed, shuffle=True)
        orig_te = set(original_rows.loc[original_rows.seed == seed, "response_id"])
        if orig_te and orig_te != set(te):
            mismatched += 1
        Xtr, ytr = f.loc[tr, FEATURES].values, f.loc[tr, "score"].values
        Xva, yva = f.loc[va, FEATURES].values, f.loc[va, "score"].values
        grid = GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]),
            {"ridge__alpha": ALPHAS},
            cv=KFold(5, shuffle=True, random_state=seed),
            scoring=scorer,
        )
        grid.fit(Xtr, ytr)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=grid.best_params_["ridge__alpha"])),
            ]
        )
        model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
        pred = clip_round(model.predict(f.loc[te, FEATURES].values))
        out.append(pd.DataFrame({"response_id": te, "seed": seed, "sw_pred": pred,
                                 "true": f.loc[te, "score"].values}))
    if mismatched:
        return None, (f"not used: held-out sets differ from the original prediction files in "
                      f"{mismatched} of {len(SEEDS)} seeds, so the comparison would not be like-for-like")
    p = pd.concat(out)
    p["abs_err"] = (p.sw_pred - p.true).abs()
    agg = p.groupby("response_id").agg(
        sliding_mean_prediction_rounded=("sw_pred", "mean"),
        sliding_mean_absolute_error=("abs_err", "mean"),
    ).reset_index()
    return agg, "used (held-out sets identical to the original prediction files)"


# ---------------------------------------------------------------------------
# 8. Run
# ---------------------------------------------------------------------------
selected_all, ranked_all, manifest, references = [], [], [], []
for task in TASKS:
    rows, pred_path = load_predictions(task)
    g = aggregate(rows, task)

    ds_path = find_dataset_file(task)
    ds = read_csv_flexible(ds_path)
    ds["response_id"] = ds.ID.map(normalized_id)
    ds["student_response"] = ds.essay.fillna("").astype(str)
    ds["student_word_count"] = ds.student_response.str.split().str.len()
    texts = ds[["response_id", "student_response", "student_word_count"]].drop_duplicates("response_id")

    sel = select_cases(g, texts)
    sw_note = "not applicable (OS)"
    if TASK_META[task][0] == "ASAP":
        sw, sw_note = sliding_predictions(task, rows)
        if sw is not None:
            sel = sel.merge(sw, on="response_id", how="left")
            rng_ = TASK_META[task][3] - TASK_META[task][2]
            sel["sliding_normalized_error"] = sel.sliding_mean_absolute_error / rng_
            sel["error_reduction_pct"] = np.where(
                sel.mean_absolute_error > 0,
                100 * (1 - sel.sliding_mean_absolute_error / sel.mean_absolute_error), np.nan)
            sel["error_persists_with_full_text"] = np.where(
                sel.error_reduction_pct >= 50, "No (error at least halved)", "Yes"
            )
    ranked_all.append(g)
    selected_all.append(sel)
    manifest.append(dict(task=task, prediction_file=str(pred_path), dataset_file=str(ds_path),
                         held_out_responses=len(g),
                         overpredicted=int((g.error_direction == "Overprediction").sum()),
                         underpredicted=int((g.error_direction == "Underprediction").sum()),
                         exact_in_all_appearances=int((g.n_exact == g.test_appearances).sum()),
                         selected=len(sel), sliding_window_check=sw_note))
    primary_references = load_primary_references(task)
    for i, row in enumerate(primary_references.itertuples(index=False), 1):
        references.append(
            {
                "task": task,
                "reference_number": i,
                "tone": getattr(row, "tone", ""),
                "reference_answer": row.reference_answer,
                "source_file": str(REFERENCES_CSV),
            }
        )
    print(f"{task}: {len(sel)} cases selected; sliding-window check {sw_note}")

sel = pd.concat(selected_all, ignore_index=True)
if "error_persists_with_full_text" in sel.columns:
    sel["error_persists_with_full_text"] = np.where(
        sel.dataset.eq("OS"), "n/a (OS, no truncation)",
        sel.error_persists_with_full_text.fillna("not checked"))
sel = add_token_counts(sel)
stratum_order = {"Extreme: overprediction": 0, "Extreme: underprediction": 1}
sel["_t"] = sel.task.map({t: i for i, t in enumerate(TASKS)})
sel["_s"] = sel.sampling_stratum.map(stratum_order)
sel = sel.sort_values(["_t", "_s", "stratum_rank"]).drop(columns=["_t", "_s"]).reset_index(drop=True)
sel.insert(0, "case_id", [f"C{i:02d}" for i in range(1, len(sel) + 1)])

coding_cols = ["R1_primary_code", "R1_secondary_code", "R1_notes",
               "R2_primary_code", "R2_secondary_code", "R2_notes",
               "consensus_primary_code", "consensus_secondary_code", "consensus_rationale",
               "representative_excerpt"]
for c in coding_cols:
    sel[c] = ""

info_cols = ["case_id", "dataset", "task", "task_description", "sampling_stratum", "stratum_rank",
             "response_id", "true_score", "mean_prediction_rounded", "mean_prediction_continuous",
             "mean_signed_error", "mean_normalized_absolute_error", "prediction_sd",
             "test_appearances", "test_seeds", "student_word_count", "sbert_tokens", "roberta_tokens",
             "exceeds_sbert_384", "exceeds_roberta_512",
             "sliding_mean_prediction_rounded", "sliding_mean_absolute_error",
             "error_reduction_pct", "error_persists_with_full_text"]
info_cols = [c for c in info_cols if c in sel.columns]
cases = sel[info_cols + ["student_response"] + coding_cols]

pd.concat(ranked_all).to_csv(OUTPUT_DIR / "all_response_error_rankings.csv", index=False)
cases.to_csv(OUTPUT_DIR / "selected_cases.csv", index=False)
pd.DataFrame(manifest).to_csv(OUTPUT_DIR / "input_file_manifest.csv", index=False)

# ---------------------------------------------------------------------------
# 9. Workbook
# ---------------------------------------------------------------------------
from openpyxl import Workbook  # noqa: E402
from openpyxl.comments import Comment  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

FONT = "Arial"
HDR = dict(font=Font(name=FONT, bold=True, color="FFFFFF"), fill=PatternFill("solid", fgColor="1F4E78"))
EDIT = PatternFill("solid", fgColor="FFF2CC")
wb = Workbook()


def write_table(ws, df, start_row=1, widths=None):
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(start_row, j, col)
        c.font, c.fill = HDR["font"], HDR["fill"]
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    for i, row in enumerate(df.itertuples(index=False), start_row + 1):
        for j, v in enumerate(row, 1):
            if isinstance(v, float) and np.isnan(v):
                v = None
            c = ws.cell(i, j, v)
            c.font = Font(name=FONT)
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for j, col in enumerate(df.columns, 1):
        ws.column_dimensions[get_column_letter(j)].width = (widths or {}).get(col, 16)
    ws.freeze_panes = ws.cell(start_row + 1, 1)


# Method sheet
ws = wb.active
ws.title = "Method"
method = [
    ("Stratified extreme-case error analysis: protocol", True),
    ("", False),
    ("1. Sampling (purposeful; Patton, 2002; Palinkas et al., 2015)", True),
    (f"Primary configuration: {GENERATOR} references, {LENGTH_CONDITION} condition, {FEATURE_SET} features; held-out predictions averaged over the seeds in which each response was in the test set.", False),
    (f"Strata: each task/question x error direction. Extreme (deviant) cases: the {EXTREME_PER_DIRECTION} most overpredicted and the {EXTREME_PER_DIRECTION} most underpredicted responses per task, ranked by mean absolute error of the rounded score divided by the score range (ties: continuous-prediction error, then number of held-out appearances).", False),
    ("", False),
    ("2. Hybrid deductive-inductive coding (Fereday & Muir-Cochrane, 2006; Hsieh & Shannon, 2005)", True),
    ("Deductive: apply the a-priori codebook (sheet Codebook), derived from the scoring rubrics and from mechanisms identified in this study.", False),
    ("Inductive: when a recurring pattern fits no code, add a new code in the empty rows of the Codebook sheet (origin = Inductive) with a definition and an example case; re-apply the updated codebook to all cases.", False),
    ("Assign one primary code (the main reason for the error) and optionally one secondary code.", False),
    ("", False),
    ("3. Reliability (Cohen, 1960; O'Connor & Joffe, 2020)", True),
    ("Two reviewers code all cases independently (R1 and R2 columns) before discussing them. Agreement on the primary code is computed in the sheet Reliability as percent agreement and Cohen's kappa. Disagreements are then resolved by discussion and recorded as consensus codes with a rationale.", False),
    ("", False),
    ("4. Reporting", True),
    ("Report the code frequencies by dataset and stratum (sheet Summary) with short excerpts for the main codes. Counts describe the selected extreme cases only and are not prevalence estimates.", False),
    ("", False),
    ("Yellow cells are to be completed by the reviewers.", False),
    ("", False),
    ("References", True),
    ("Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37-46.", False),
    ("Fereday, J., & Muir-Cochrane, E. (2006). Demonstrating rigor using thematic analysis: A hybrid approach of inductive and deductive coding and theme development. International Journal of Qualitative Methods, 5(1), 80-92.", False),
    ("Hsieh, H.-F., & Shannon, S. E. (2005). Three approaches to qualitative content analysis. Qualitative Health Research, 15(9), 1277-1288.", False),
    ("O'Connor, C., & Joffe, H. (2020). Intercoder reliability in qualitative research: Debates and practical guidelines. International Journal of Qualitative Methods, 19, 1-13.", False),
    ("Palinkas, L. A., Horwitz, S. M., Green, C. A., Wisdom, J. P., Duan, N., & Hoagwood, K. (2015). Purposeful sampling for qualitative data collection and analysis in mixed method implementation research. Administration and Policy in Mental Health, 42(5), 533-544.", False),
    ("Patton, M. Q. (2002). Qualitative research and evaluation methods (3rd ed.). Sage.", False),
]
for i, (text, bold) in enumerate(method, 1):
    c = ws.cell(i, 1, text)
    c.font = Font(name=FONT, bold=bold, size=12 if i == 1 else 10)
    c.alignment = Alignment(wrap_text=True, vertical="top")
ws.column_dimensions["A"].width = 130

# Codebook sheet
cb = wb.create_sheet("Codebook")
cbdf = pd.DataFrame(CODEBOOK, columns=["code", "group", "definition", "basis"])
cbdf.insert(2, "origin", "Deductive (a priori)")
cbdf["example_case_id"] = ""
write_table(cb, cbdf, widths={"code": 36, "group": 18, "origin": 20, "definition": 70, "basis": 34,
                              "example_case_id": 16})
for r in range(len(cbdf) + 2, CODEBOOK_ROWS + 2):
    for j in range(1, 7):
        cb.cell(r, j).fill = EDIT
        cb.cell(r, j).font = Font(name=FONT)
cb.cell(len(cbdf) + 2, 3).value = "Inductive"
cb.cell(len(cbdf) + 2, 3).comment = Comment("Add new inductive codes in these yellow rows.", "AES")
code_range = f"Codebook!$A$2:$A${CODEBOOK_ROWS + 1}"

# Cases sheet
cs = wb.create_sheet("Cases")
widths = {c: 14 for c in cases.columns}
widths.update({"task_description": 26, "sampling_stratum": 22, "student_response": 80,
               "R1_notes": 40, "R2_notes": 40, "consensus_rationale": 45, "representative_excerpt": 45,
               "error_persists_with_full_text": 20})
for c in coding_cols:
    if c.endswith("code"):
        widths[c] = 30
write_table(cs, cases, widths=widths)
n = len(cases)
dv = DataValidation(type="list", formula1=f"={code_range}", allow_blank=True)
dv.errorTitle, dv.error = "Unknown code", "Choose a code from the Codebook sheet (add new codes there first)."
cs.add_data_validation(dv)
for c in coding_cols:
    j = list(cases.columns).index(c) + 1
    col = get_column_letter(j)
    for r in range(2, n + 2):
        cs.cell(r, j).fill = EDIT
    if c.endswith("code"):
        dv.add(f"{col}2:{col}{n + 1}")
for r in range(2, n + 2):
    cs.row_dimensions[r].height = 120

col_of = {c: get_column_letter(list(cases.columns).index(c) + 1) for c in cases.columns}
R1 = f"Cases!${col_of['R1_primary_code']}$2:${col_of['R1_primary_code']}${n + 1}"
R2 = f"Cases!${col_of['R2_primary_code']}$2:${col_of['R2_primary_code']}${n + 1}"
CON = f"Cases!${col_of['consensus_primary_code']}$2:${col_of['consensus_primary_code']}${n + 1}"
DS = f"Cases!${col_of['dataset']}$2:${col_of['dataset']}${n + 1}"
STR = f"Cases!${col_of['sampling_stratum']}$2:${col_of['sampling_stratum']}${n + 1}"

# Reliability sheet (live formulas)
rl = wb.create_sheet("Reliability")
rl["A1"] = "Inter-rater agreement on the primary code (R1 vs R2)"
rl["A1"].font = Font(name=FONT, bold=True, size=12)
labels = [("Cases coded by both reviewers", f'=SUMPRODUCT(({R1}<>"")*({R2}<>""))'),
          ("Cases with identical primary code", f'=SUMPRODUCT(({R1}<>"")*({R1}={R2}))'),
          ("Percent agreement (observed, Po)", "=IFERROR(B4/B3,\"\")"),
          ("Expected agreement by chance (Pe)", f"=IFERROR(SUM(D11:D{10 + CODEBOOK_ROWS}),\"\")"),
          ("Cohen's kappa", "=IFERROR((B5-B6)/(1-B6),\"\")")]
for i, (lab, f) in enumerate(labels, 3):
    rl.cell(i, 1, lab).font = Font(name=FONT)
    c = rl.cell(i, 2, f)
    c.font = Font(name=FONT, bold=(i == 7))
    c.number_format = "0.0%" if i == 5 else "0.000"
rl["B6"].number_format = "0.000"
rl["A8"] = "Kappa is computed over cases coded by both reviewers. With few cases, report it together with percent agreement."
rl["A8"].font = Font(name=FONT, italic=True, size=9)
for j, h in enumerate(["Code", "R1 count", "R2 count", "Chance term"], 1):
    c = rl.cell(10, j, h)
    c.font, c.fill = HDR["font"], HDR["fill"]
for k in range(CODEBOOK_ROWS):
    r = 11 + k
    rl.cell(r, 1, f'=IF(Codebook!A{2 + k}="","",Codebook!A{2 + k})')
    rl.cell(r, 2, f'=IF(A{r}="","",SUMPRODUCT(({R1}=A{r})*({R2}<>"")))')
    rl.cell(r, 3, f'=IF(A{r}="","",SUMPRODUCT(({R2}=A{r})*({R1}<>"")))')
    rl.cell(r, 4, f'=IF(OR(A{r}="",$B$3=0),0,(B{r}/$B$3)*(C{r}/$B$3))')
    for j in range(1, 5):
        rl.cell(r, j).font = Font(name=FONT)
rl.column_dimensions["A"].width = 44
for col in "BCD":
    rl.column_dimensions[col].width = 14

# Summary sheet (live formulas on consensus primary codes)
sm = wb.create_sheet("Summary")
sm["A1"] = "Consensus primary codes by dataset and sampling stratum (counts of selected cases)"
sm["A1"].font = Font(name=FONT, bold=True, size=12)
strata = list(stratum_order)
heads = ["Code"] + [f"{d}: {s}" for d in ("ASAP", "OS") for s in strata] + ["Total"]
for j, h in enumerate(heads, 1):
    c = sm.cell(3, j, h)
    c.font, c.fill = HDR["font"], HDR["fill"]
    c.alignment = Alignment(wrap_text=True, horizontal="center")
for k in range(CODEBOOK_ROWS):
    r = 4 + k
    sm.cell(r, 1, f'=IF(Codebook!A{2 + k}="","",Codebook!A{2 + k})').font = Font(name=FONT)
    j = 2
    for d in ("ASAP", "OS"):
        for s in strata:
            sm.cell(r, j, f'=IF($A{r}="","",COUNTIFS({CON},$A{r},{DS},"{d}",{STR},"{s}"))').font = Font(name=FONT)
            j += 1
    first, last = get_column_letter(2), get_column_letter(j - 1)
    sm.cell(r, j, f'=IF($A{r}="","",SUM({first}{r}:{last}{r}))').font = Font(name=FONT)
tr_ = 4 + CODEBOOK_ROWS
sm.cell(tr_, 1, "Cases selected").font = Font(name=FONT, bold=True)
j = 2
for d in ("ASAP", "OS"):
    for s in strata:
        sm.cell(tr_, j, f'=COUNTIFS({DS},"{d}",{STR},"{s}")').font = Font(name=FONT, bold=True)
        j += 1
sm.cell(tr_, j, f"=SUM(B{tr_}:{get_column_letter(j - 1)}{tr_})").font = Font(name=FONT, bold=True)
sm.column_dimensions["A"].width = 40
for k in range(2, j + 1):
    sm.column_dimensions[get_column_letter(k)].width = 16

# References, configuration and input files
write_table(wb.create_sheet("References_GPT_long"), pd.DataFrame(references),
            widths={"reference_answer": 90, "source_file": 50})
cfg = pd.DataFrame([
    ("Generator / condition / features", f"{GENERATOR} / {LENGTH_CONDITION} / {FEATURE_SET}"),
    ("Extreme cases per task and direction", EXTREME_PER_DIRECTION),
    ("Minimum held-out appearances", MIN_APPEARANCES),
    ("Error measure", "Mean |rounded prediction - human score| / score range, over held-out appearances"),
    ("Sliding-window features", str(SLIDING_RUN)),
    ("'Error persists' rule", "Yes unless the full-text (sliding-window) error is at least 50% smaller"),
    ("Run", RUN_STAMP),
], columns=["setting", "value"])
write_table(wb.create_sheet("Configuration"), cfg, widths={"setting": 38, "value": 90})
write_table(wb.create_sheet("Input_Files"), pd.DataFrame(manifest),
            widths={"prediction_file": 60, "dataset_file": 50, "sliding_window_check": 60})

wb_path = OUTPUT_DIR / "AES_error_case_review_workbook_v2.xlsx"
wb.save(wb_path)

counts = cases.groupby(["dataset", "sampling_stratum"]).size().unstack(fill_value=0)
print("\nSelected cases by dataset and stratum:")
print(counts.to_string())
print("\nCases:")
show = ["case_id", "task", "sampling_stratum", "response_id", "true_score", "mean_prediction_rounded",
        "student_word_count", "exceeds_sbert_384", "error_persists_with_full_text"]
print(cases[[c for c in show if c in cases.columns]].to_string(index=False))
print(f"\nWorkbook: {wb_path}")
