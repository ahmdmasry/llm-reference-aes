"""Truncation-sensitivity analysis for the ASAP tasks affected by encoder limits.

Reviewer comment: "Conduct a tokenizer-length audit and truncation sensitivity
analysis ... Re-run key analyses with chunking, sliding windows, or long-context
encoders. This is essential because the ASAP length effect may be partly an
artifact of truncation."

Scope: all eight ASAP tasks. Responses or references exceed the encoder limits
in every ASAP task (in ASAP 3 to 6 mainly the long references). The OS
questions are not re-run: none of their references and at most 1% of their
responses exceed either limit.

Method (sliding-window context, token-level aggregation):
  * Every text is tokenized in full with the encoder's own tokenizer.
  * Texts that fit the encoder are encoded exactly as in the original pipeline.
    Longer texts are split into overlapping windows of the maximum length
    (382 content tokens for all-mpnet-base-v2, 510 for roberta-large); each
    window gets its own special tokens, and every token keeps the embedding
    from the window in which it is most central.
  * SBERT: the document vector is the mean of all token embeddings (the
    model's own mean pooling) followed by L2 normalization.
  * BERTScore: greedy cosine matching over the full token sets of both
    documents, layer 17 of roberta-large, no idf, no baseline rescaling,
    exactly as in the bert-score package, but without truncation.
  For texts that fit in one window, both procedures reproduce the original
  features; this is verified against the stored features before scoring.

Scoring repeats the paper's protocol on the original (direct-truncation)
features and on the sliding-window features with identical splits, seeds and
lambda grid, and reports how the short/medium/long gaps change.

Nothing in the existing feature or result folders is modified. Outputs are
written to a new time-stamped directory. A CUDA GPU is strongly recommended.
"""

import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers, sklearn, scipy
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FEATURE_ROOT = Path(
    os.environ.get('AES_FEATURE_ROOT', PACKAGE_ROOT / 'artifacts' / 'numeric_features')
)
DATA_DIR = Path(os.environ.get('AES_DATA_DIR', PACKAGE_ROOT / 'data'))
REFERENCES_CSV = Path(
    os.environ.get(
        'AES_REFERENCES_CSV',
        PACKAGE_ROOT / 'artifacts' / 'generated_references' / 'all_generated_references.csv',
    )
)
OUT_ROOT = Path(
    os.environ.get(
        'AES_TRUNCATION_OUTPUT_DIR',
        PACKAGE_ROOT / 'local_outputs' / 'truncation_sensitivity',
    )
)
RUN_STAMP = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')

TASKS = {               # official score ranges as reported in Table 1
    'ASAP1': (2, 12),
    'ASAP2': (2, 10),
    'ASAP3': (0, 3),
    'ASAP4': (0, 3),
    'ASAP5': (0, 4),
    'ASAP6': (0, 4),
    'ASAP7': (0, 30),
    'ASAP8': (0, 60),
}
GENERATORS = ('GPT', 'Gemini', 'Claude')
LENGTHS = ('short', 'medium', 'long')
FEATURES = ['similarity_mean', 'bert_P_mean', 'bert_R_mean', 'bert_F1_mean']
SEEDS = (42, 43, 44, 45, 46)
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

SBERT_MODEL = 'sentence-transformers/all-mpnet-base-v2'
BERT_MODEL = 'roberta-large'
BERT_LAYER = 17                 # bert-score default for roberta-large
SBERT_OVERLAP = 128             # tokens shared by consecutive windows
BERT_OVERLAP = 128
ESSAY_BLOCK = 128               # essays encoded per block (memory control)
WINDOW_BATCH = 16               # windows per forward pass

# Equivalence check against the stored features for texts that fit in one
# window: differences above EQ_WARN are reported, above EQ_FAIL stop the run.
EQ_WARN, EQ_FAIL = 1e-4, 5e-3

# Seed-averaged QWK reported in Table 4 (combined features), used to confirm
# that the direct-truncation results reproduce the paper.
PAPER_QWK = {
    'ASAP1': {'GPT': (.584, .678, .712), 'Gemini': (.547, .637, .709), 'Claude': (.558, .685, .737)},
    'ASAP2': {'GPT': (.566, .684, .721), 'Gemini': (.503, .667, .669), 'Claude': (.515, .706, .729)},
    'ASAP3': {'GPT': (.528, .550, .574), 'Gemini': (.481, .563, .586), 'Claude': (.506, .562, .576)},
    'ASAP4': {'GPT': (.669, .694, .706), 'Gemini': (.622, .672, .694), 'Claude': (.631, .670, .686)},
    'ASAP5': {'GPT': (.727, .762, .763), 'Gemini': (.701, .728, .737), 'Claude': (.706, .750, .745)},
    'ASAP6': {'GPT': (.753, .786, .778), 'Gemini': (.762, .784, .779), 'Claude': (.739, .752, .743)},
    'ASAP7': {'GPT': (.613, .733, .738), 'Gemini': (.470, .584, .636), 'Claude': (.597, .671, .700)},
    'ASAP8': {'GPT': (.276, .556, .597), 'Gemini': (.364, .578, .624), 'Claude': (.337, .564, .594)},
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE != 'cuda':
    print('WARNING: no GPU detected; the run will be very slow.')
torch.backends.cuda.matmul.allow_tf32 = False   # keep fp32 exactness for the
torch.backends.cudnn.allow_tf32 = False         # equivalence check


# I/O helpers
def read_csv_any(path):
    for enc in ('utf-8-sig', 'cp1252', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode {path}')


RUN_DIR = OUT_ROOT / RUN_STAMP
if RUN_DIR.exists():
    raise FileExistsError(f'{RUN_DIR} already exists; refusing to overwrite.')
RUN_DIR.mkdir(parents=True)


def save_csv(df, name):
    path = RUN_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    df.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def find_dataset(task):
    number = task.removeprefix('ASAP')
    candidates = (
        DATA_DIR / f'{task}.csv',
        DATA_DIR / f'{task.lower()}.csv',
        DATA_DIR / f'ASAP_task{number}.csv',
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f'No dataset CSV found for {task}. Place it in {DATA_DIR} or set AES_DATA_DIR.'
        )
    return path


def load_baseline(task):
    """Load public numerical features and merge the independently obtained text."""
    source = read_csv_any(find_dataset(task))
    missing = {'ID', 'essay', 'score'} - set(source.columns)
    if missing:
        raise ValueError(f'{task} dataset lacks {sorted(missing)}')
    ref = (source[['ID', 'essay', 'score']]
           .dropna(subset=['score']).drop_duplicates('ID')
           .sort_values('ID').reset_index(drop=True))
    frames = {}
    for g in GENERATORS:
        for l in LENGTHS:
            path = FEATURE_ROOT / task / g / f'features_full_{l}_{task}_{g}.csv'
            df = read_csv_any(path)
            df.columns = [str(c).replace('﻿', '').replace('ï»¿', '').strip() for c in df.columns]
            df = df.sort_values('ID').reset_index(drop=True)
            missing = set(FEATURES + ['ID', 'score']) - set(df.columns)
            if missing:
                raise ValueError(f'{path.name} lacks {sorted(missing)}')
            if not (np.array_equal(df.ID, ref.ID) and np.array_equal(df.score, ref.score)):
                raise ValueError(f'{path.name}: IDs or scores differ from the dataset')
            frames[(g, l)] = ref.merge(
                df[['ID', *FEATURES]], on='ID', validate='one_to_one'
            )
    ref['essay'] = ref.essay.astype(str)
    ref['score'] = ref.score.astype(int)
    lo, hi = TASKS[task]
    if ref.score.min() < lo or ref.score.max() > hi:
        raise ValueError(f'{task}: scores outside {lo}-{hi}')
    return ref, frames


def load_references(task):
    frame = read_csv_any(REFERENCES_CSV)
    required = {'task', 'generator', 'condition', 'reference_answer'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'{REFERENCES_CSV} lacks {sorted(missing)}')
    frame = frame[frame.task.astype(str).str.casefold().eq(task.casefold())]
    refs = {}
    for g in GENERATORS:
        for l in LENGTHS:
            selected = frame[
                frame.generator.astype(str).str.casefold().eq(g.casefold())
                & frame.condition.astype(str).str.casefold().eq(l)
            ]
            texts = [
                text.strip()
                for text in selected.reference_answer.dropna().astype(str)
                if text.strip()
            ]
            if len(texts) != 5:
                raise ValueError(f'{task}/{g}/{l}: expected 5 references, found {len(texts)}')
            refs[(g, l)] = texts
    return refs


# Sliding-window token encoder
class WindowEncoder:
    """Encodes documents of any length with a fixed-context encoder.

    Returns, per document, the token embeddings of its content tokens plus the
    embeddings of the opening and closing special tokens. A document that fits
    in one window is encoded exactly as a standard truncating pipeline would.
    """

    def __init__(self, model, tokenizer, window, overlap, tok_kwargs=None):
        self.model, self.tok = model.eval(), tokenizer
        self.W, self.ov = window, overlap
        self.stride = window - overlap
        self.tok_kwargs = tok_kwargs or {}
        # Opening and closing special tokens (<s> ... </s> for both encoders).
        # Added explicitly because recent transformers versions removed
        # build_inputs_with_special_tokens from some tokenizers.
        self.cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        probe = tokenizer('probe text', add_special_tokens=True)['input_ids']
        if probe[0] != self.cls_id or probe[-1] != self.sep_id:
            raise RuntimeError(f'Unexpected special-token layout for {type(tokenizer).__name__}: {probe}')

    def ids(self, text):
        return self.tok.encode(text.strip(), add_special_tokens=False,
                               truncation=False, **self.tok_kwargs)

    def starts(self, n):
        if n <= self.W:
            return [0]
        out, s = [], 0
        while True:
            out.append(s)
            if s + self.W >= n:
                return out
            s += self.stride

    @torch.no_grad()
    def encode(self, texts):
        plans, windows = [], []
        for d, text in enumerate(texts):
            ids = self.ids(text)
            st = self.starts(len(ids))
            plans.append((len(ids), st, len(windows)))
            windows += [ids[s:s + self.W] for s in st]
        seqs = [[self.cls_id] + w + [self.sep_id] for w in windows]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        hidden = [None] * len(seqs)
        pad = self.tok.pad_token_id
        for b in range(0, len(order), WINDOW_BATCH):
            idx = order[b:b + WINDOW_BATCH]
            L = max(len(seqs[i]) for i in idx)
            input_ids = torch.full((len(idx), L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                input_ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                att[j, :len(seqs[i])] = 1
            out = self.model(input_ids=input_ids.to(DEVICE), attention_mask=att.to(DEVICE))
            H = out.last_hidden_state.float()
            for j, i in enumerate(idx):
                hidden[i] = H[j, :len(seqs[i])].cpu()
        docs = []
        for n, st, w0 in plans:
            wins = hidden[w0:w0 + len(st)]
            parts = []
            for k, s in enumerate(st):
                lo = 0 if k == 0 else s + self.ov // 2
                hi = n if k == len(st) - 1 else st[k + 1] + self.ov // 2
                parts.append(wins[k][1 + lo - s:1 + hi - s])
            content = torch.cat(parts) if n else wins[0][1:1]
            docs.append(dict(content=content, cls=wins[0][0], sep=wins[-1][-1],
                             n_tokens=n, n_windows=len(st)))
        return docs


def sbert_vectors(docs):
    """Mean pooling over all tokens (including special tokens), then L2
    normalization, as in the all-mpnet-base-v2 pipeline."""
    vecs = []
    for d in docs:
        X = torch.cat([d['cls'][None], d['content'], d['sep'][None]])
        v = X.mean(0)
        vecs.append(v / v.norm())
    return torch.stack(vecs)


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def bert_views(d):
    """Scoring tokens (content) and matching tokens (content plus special
    tokens), as in bert-score's greedy matching with idf disabled."""
    content = _norm(d['content'])
    full = _norm(torch.cat([d['cls'][None], d['content'], d['sep'][None]]))
    return content, full


def bertscore_against(e_c, e_f, ref_pack):
    """Precision, recall and F1 of one essay (candidate) against each
    reference of one condition, using full token sets. e_c and e_f are the
    essay's normalized content and full token embeddings on the device."""
    sim_p = e_c @ ref_pack['full'].T          # essay content x ref (all tokens)
    sim_r = e_f @ ref_pack['content'].T       # essay (all tokens) x ref content
    P, R = [], []
    for (a, b), (c, d) in zip(ref_pack['seg_full'], ref_pack['seg_content']):
        P.append(sim_p[:, a:b].max(dim=1).values.mean().item())
        R.append(sim_r[:, c:d].max(dim=0).values.mean().item())
    P, R = np.array(P), np.array(R)
    F = 2 * P * R / (P + R + 1e-12)
    return P, R, F


def pack_refs(ref_docs):
    contents, fulls, seg_c, seg_f, pc, pf = [], [], [], [], 0, 0
    for d in ref_docs:
        c, f = bert_views(d)
        contents.append(c); fulls.append(f)
        seg_c.append((pc, pc + len(c))); pc += len(c)
        seg_f.append((pf, pf + len(f))); pf += len(f)
    return dict(content=torch.cat(contents).to(DEVICE), full=torch.cat(fulls).to(DEVICE),
                seg_content=seg_c, seg_full=seg_f)


# Load encoders
print('Loading encoders ...')
sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)
SBERT_WINDOW = sbert.max_seq_length - 2           # 382 content tokens
sbert_enc = WindowEncoder(sbert[0].auto_model, sbert[0].tokenizer, SBERT_WINDOW, SBERT_OVERLAP)

bert_model = AutoModel.from_pretrained(BERT_MODEL)
bert_model.encoder.layer = torch.nn.ModuleList(list(bert_model.encoder.layer)[:BERT_LAYER])
bert_model.to(DEVICE)
BERT_WINDOW = 510                                 # 512 minus the two special tokens


def make_bert_encoder(prefix_space):
    tok = AutoTokenizer.from_pretrained(BERT_MODEL, use_fast=False, add_prefix_space=prefix_space)
    return WindowEncoder(bert_model, tok, BERT_WINDOW, BERT_OVERLAP)


def calibrate_bert_tokenization(task='ASAP1', generator='GPT', length='short', n_essays=150):
    """Whether bert-score adds a leading space before the first word depends on
    the transformers version it ran with. Both settings are tested on essays and
    references that fit in one window, and the one that reproduces the stored
    BERTScore features is kept."""
    essays, baseline = load_baseline(task)
    refs = load_references(task)[(generator, length)]
    base = baseline[(generator, length)]
    results = {}
    for prefix in (False, True):
        enc = make_bert_encoder(prefix)
        idx = [i for i, t in enumerate(essays.essay) if len(enc.ids(t)) <= BERT_WINDOW][:n_essays]
        rdocs = enc.encode(refs)
        if any(d['n_windows'] > 1 for d in rdocs):
            raise RuntimeError('Calibration references must fit in one window')
        pack = pack_refs(rdocs)
        edocs = enc.encode(essays.essay.iloc[idx].tolist())
        P, R = [], []
        for d in edocs:
            p_, r_, _ = bertscore_against(*(v.to(DEVICE) for v in bert_views(d)), pack)
            P.append(p_.mean()); R.append(r_.mean())
        diff = max(np.abs(np.array(P) - base.bert_P_mean.values[idx]).max(),
                   np.abs(np.array(R) - base.bert_R_mean.values[idx]).max())
        results[prefix] = diff
        print(f'  add_prefix_space={prefix}: max |difference| = {diff:.2e} over {len(idx)} essays')
    best = min(results, key=results.get)
    if results[best] > EQ_FAIL:
        raise RuntimeError(f'Neither tokenization setting reproduces the stored BERTScore '
                           f'features ({results}). The original features may have been computed '
                           'with different settings; check the original extraction code.')
    return best, results


print('Calibrating RoBERTa tokenization against the stored features ...')
BERT_PREFIX_SPACE, CALIBRATION = calibrate_bert_tokenization()
bert_enc = make_bert_encoder(BERT_PREFIX_SPACE)
print(f'Using add_prefix_space={BERT_PREFIX_SPACE}')
print(f'SBERT window {SBERT_WINDOW}, BERTScore window {BERT_WINDOW}, overlaps {SBERT_OVERLAP}/{BERT_OVERLAP}')


# Sliding-window features for each task
def sliding_features(task, essays_df, refs):
    t0 = time.time()
    ref_info, packs, ref_vecs = [], {}, {}
    for (g, l), texts in refs.items():
        sd, bd = sbert_enc.encode(texts), bert_enc.encode(texts)
        ref_vecs[(g, l)] = sbert_vectors(sd).to(DEVICE)
        packs[(g, l)] = pack_refs(bd)
        for i, (a, b) in enumerate(zip(sd, bd)):
            ref_info.append(dict(task=task, generator=g, length=l, reference=i + 1,
                                 sbert_tokens=a['n_tokens'], sbert_windows=a['n_windows'],
                                 roberta_tokens=b['n_tokens'], roberta_windows=b['n_windows']))
    rows = {k: [] for k in refs}
    essay_info = []
    texts = essays_df.essay.tolist()
    for b0 in range(0, len(texts), ESSAY_BLOCK):
        block = texts[b0:b0 + ESSAY_BLOCK]
        sd, bd = sbert_enc.encode(block), bert_enc.encode(block)
        ev = sbert_vectors(sd).to(DEVICE)
        for j in range(len(block)):
            essay_info.append(dict(sbert_tokens=sd[j]['n_tokens'], sbert_windows=sd[j]['n_windows'],
                                   roberta_tokens=bd[j]['n_tokens'], roberta_windows=bd[j]['n_windows']))
        views = [tuple(v.to(DEVICE) for v in bert_views(d)) for d in bd]
        for key in refs:
            cos = (ev @ ref_vecs[key].T).cpu().numpy()
            for j in range(len(block)):
                P, R, F = bertscore_against(*views[j], packs[key])
                rows[key].append(dict(similarity_mean=cos[j].mean(), similarity_max=cos[j].max(),
                                      bert_P_mean=P.mean(), bert_R_mean=R.mean(),
                                      bert_F1_mean=F.mean(), bert_F1_max=F.max()))
        print(f'  {task}: {min(b0 + ESSAY_BLOCK, len(texts))}/{len(texts)} essays '
              f'({(time.time() - t0) / 60:.1f} min)')
    info = pd.DataFrame(essay_info)
    out = {}
    for key, r in rows.items():
        df = pd.concat([essays_df.reset_index(drop=True), pd.DataFrame(r), info], axis=1)
        df['task'], df['generator'], df['length'] = task, key[0], key[1]
        df['method'] = 'sliding_window'
        out[key] = df
    return out, info, pd.DataFrame(ref_info)


def equivalence_check(task, baseline, sliding, essay_info, ref_info):
    """Texts that fit in one window must reproduce the stored features."""
    rows = []
    fits_s = (essay_info.sbert_windows == 1).values
    fits_b = (essay_info.roberta_windows == 1).values
    for (g, l), base in baseline.items():
        ri = ref_info[(ref_info.generator == g) & (ref_info.length == l)]
        groups = [('similarity_mean', fits_s, (ri.sbert_windows == 1).all())]
        groups += [(f, fits_b, (ri.roberta_windows == 1).all())
                   for f in ('bert_P_mean', 'bert_R_mean', 'bert_F1_mean')]
        for feat, mask, refs_fit in groups:
            if not refs_fit or mask.sum() == 0:
                continue
            diff = np.abs(sliding[(g, l)][feat].values[mask] - base[feat].values[mask])
            rows.append(dict(task=task, generator=g, length=l, feature=feat,
                             n_compared=int(mask.sum()), max_abs_diff=float(diff.max()),
                             mean_abs_diff=float(diff.mean())))
    return pd.DataFrame(rows)


# Scoring
def splits(ids, seed):
    tr, tmp = train_test_split(ids, test_size=0.4, random_state=seed, shuffle=True)
    va, te = train_test_split(tmp, test_size=0.5, random_state=seed, shuffle=True)
    return tr, va, te


def fit_and_score(frame, tr, va, te, seed, lo, hi):
    f = frame.set_index('ID')
    def clip_round(x):
        return np.clip(np.rint(x), lo, hi).astype(int)
    qwk_scorer = make_scorer(lambda y, p: cohen_kappa_score(y, clip_round(p), weights='quadratic'))
    Xtr, ytr = f.loc[tr, FEATURES].values, f.loc[tr, 'score'].values
    Xva, yva = f.loc[va, FEATURES].values, f.loc[va, 'score'].values
    Xte, yte = f.loc[te, FEATURES].values, f.loc[te, 'score'].values
    grid = GridSearchCV(
        Pipeline([('scaler', StandardScaler()), ('ridge', Ridge())]),
        {'ridge__alpha': ALPHAS},
        cv=KFold(5, shuffle=True, random_state=seed),
        scoring=qwk_scorer,
    )
    grid.fit(Xtr, ytr)
    alpha = grid.best_params_['ridge__alpha']
    model = Pipeline(
        [('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))]
    )
    model.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    raw = model.predict(Xte)
    pred = clip_round(raw)
    return dict(alpha=alpha, qwk=cohen_kappa_score(yte, pred, weights='quadratic'),
                pearson_r=float(np.corrcoef(yte, raw)[0, 1]),
                mae=float(mean_absolute_error(yte, raw)))


def safe_wilcoxon(x):
    x = np.asarray(x, float)
    if len(x) < 5 or np.allclose(x, 0):
        return np.nan, np.nan
    try:
        s, p = wilcoxon(x, zero_method='wilcox')
        return float(s), float(p)
    except ValueError:
        return np.nan, np.nan


# Run
manifest = dict(
    created_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(RUN_DIR),
    tasks=TASKS, generators=GENERATORS, lengths=LENGTHS, features=FEATURES,
    seeds=SEEDS, alphas=ALPHAS, scaler_inside_cv=True,
    sbert_model=SBERT_MODEL, sbert_window=SBERT_WINDOW, sbert_overlap=SBERT_OVERLAP,
    bertscore_model=BERT_MODEL, bertscore_layer=BERT_LAYER, bertscore_window=BERT_WINDOW,
    bertscore_overlap=BERT_OVERLAP, bertscore_idf=False, bertscore_rescale=False,
    bertscore_add_prefix_space=BERT_PREFIX_SPACE, tokenization_calibration_max_diff=CALIBRATION,
    method=('token-level sliding-window context; each token keeps the embedding from the '
            'window where it is most central; SBERT mean pooling and BERTScore greedy matching '
            'over full token sets'),
    device=DEVICE, gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
    python=platform.python_version(), torch=torch.__version__,
    transformers=transformers.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
)
(RUN_DIR / 'run_manifest.json').write_text(
    json.dumps(manifest, indent=2, default=str), encoding='utf-8'
)

eq_all, extent_rows, ref_audit, score_rows = [], [], [], []
for task, (lo, hi) in TASKS.items():
    print(f'\n=== {task} ===')
    essays, baseline = load_baseline(task)
    refs = load_references(task)
    sliding, essay_info, ref_info = sliding_features(task, essays, refs)

    for (g, l), df in sliding.items():
        save_csv(df, f'{task}/features_sliding_{l}_{task}_{g}.csv')
    ref_audit.append(ref_info)
    extent_rows.append(dict(
        task=task, n_responses=len(essays),
        responses_over_sbert_pct=100 * (essay_info.sbert_windows > 1).mean(),
        responses_over_roberta_pct=100 * (essay_info.roberta_windows > 1).mean(),
        references_over_sbert_pct=100 * (ref_info.sbert_windows > 1).mean(),
        references_over_roberta_pct=100 * (ref_info.roberta_windows > 1).mean(),
        max_sbert_windows=int(max(essay_info.sbert_windows.max(), ref_info.sbert_windows.max())),
        max_roberta_windows=int(max(essay_info.roberta_windows.max(), ref_info.roberta_windows.max())),
    ))

    eq = equivalence_check(task, baseline, sliding, essay_info, ref_info)
    eq_all.append(eq)
    worst = eq.max_abs_diff.max() if len(eq) else np.nan
    print(f'  Equivalence check: max |difference| = {worst:.2e} over {len(eq)} feature sets')
    if len(eq) and worst > EQ_FAIL:
        save_csv(pd.concat(eq_all), 'equivalence_check.csv')
        raise RuntimeError(
            f'{task}: sliding-window features do not reproduce the stored features for texts that '
            f'fit in one window (max diff {worst:.3g}). The two pipelines differ in more than '
            'truncation; see equivalence_check.csv before interpreting any result.')
    if len(eq) and worst > EQ_WARN:
        print('  WARNING: small differences above tolerance; see equivalence_check.csv')

    ids = essays.ID.values
    for seed in SEEDS:
        tr, va, te = splits(ids, seed)
        for g in GENERATORS:
            for l in LENGTHS:
                for method, frame in (('direct_truncation', baseline[(g, l)]),
                                      ('sliding_window', sliding[(g, l)])):
                    r = fit_and_score(frame, tr, va, te, seed, lo, hi)
                    score_rows.append(dict(task=task, generator=g, length=l, method=method,
                                           seed=seed, **r))
    print(f'  Scoring complete for {task}')

# Summaries
raw = pd.DataFrame(score_rows)
save_csv(raw, 'scoring_raw.csv')
save_csv(pd.concat(eq_all), 'equivalence_check.csv')
save_csv(pd.DataFrame(extent_rows).round(2), 'truncation_extent.csv')
save_csv(pd.concat(ref_audit), 'reference_token_windows.csv')

summary = (raw.groupby(['task', 'generator', 'length', 'method'])
              .agg(qwk_mean=('qwk', 'mean'), qwk_sd=('qwk', 'std'),
                   r_mean=('pearson_r', 'mean'), mae_mean=('mae', 'mean')).reset_index())
save_csv(summary.round(4), 'scoring_summary.csv')

# Reproduction of Table 4 by the direct-truncation features
rep = summary[summary.method == 'direct_truncation'].copy()
rep['paper_qwk'] = [PAPER_QWK[t][g][LENGTHS.index(l)] for t, g, l in zip(rep.task, rep.generator, rep.length)]
rep['difference'] = rep.qwk_mean - rep.paper_qwk
save_csv(rep[['task', 'generator', 'length', 'qwk_mean', 'paper_qwk', 'difference']].round(4),
         'reproduction_check.csv')

# Sliding window versus direct truncation, per condition
wide = raw.pivot_table(index=['task', 'generator', 'length', 'seed'], columns='method',
                       values='qwk').reset_index()
wide['delta'] = wide.sliding_window - wide.direct_truncation
cmp_rows = []
for (t, l), g in wide.groupby(['task', 'length']):
    s, p = safe_wilcoxon(g.delta)
    cmp_rows.append(dict(task=t, length=l, n_pairs=len(g),
                         direct_qwk=g.direct_truncation.mean(), sliding_qwk=g.sliding_window.mean(),
                         delta_mean=g.delta.mean(), delta_sd=g.delta.std(),
                         wilcoxon_stat=s, p_value=p))
for l, g in wide.groupby('length'):
    s, p = safe_wilcoxon(g.delta)
    cmp_rows.append(dict(task='all tasks', length=l, n_pairs=len(g),
                         direct_qwk=g.direct_truncation.mean(), sliding_qwk=g.sliding_window.mean(),
                         delta_mean=g.delta.mean(), delta_sd=g.delta.std(),
                         wilcoxon_stat=s, p_value=p))
save_csv(pd.DataFrame(cmp_rows).round(4), 'method_comparison.csv')
save_csv(wide.round(4), 'method_comparison_seedwise.csv')

# Does the length effect survive? Gaps between conditions under each method
gaps = []
for (t, g, seed, m), d in raw.groupby(['task', 'generator', 'seed', 'method']):
    q = d.set_index('length').qwk
    for name, a, b in (('medium_minus_short', 'medium', 'short'),
                       ('long_minus_medium', 'long', 'medium'),
                       ('long_minus_short', 'long', 'short')):
        gaps.append(dict(task=t, generator=g, seed=seed, method=m, contrast=name, gap=q[a] - q[b]))
gaps = pd.DataFrame(gaps).pivot_table(index=['task', 'generator', 'seed', 'contrast'],
                                      columns='method', values='gap').reset_index()
gaps['change'] = gaps.sliding_window - gaps.direct_truncation
save_csv(gaps.round(4), 'length_effect_seedwise.csv')

le_rows = []
for keys, d in list(gaps.groupby(['task', 'contrast'])) + \
        [(('all tasks', c), d) for c, d in gaps.groupby('contrast')]:
    s1, p_dir = safe_wilcoxon(d.direct_truncation)
    s2, p_sw = safe_wilcoxon(d.sliding_window)
    s3, p_chg = safe_wilcoxon(d.change)
    le_rows.append(dict(task=keys[0], contrast=keys[1], n_pairs=len(d),
                        gap_direct=d.direct_truncation.mean(), p_gap_direct=p_dir,
                        gap_sliding=d.sliding_window.mean(), p_gap_sliding=p_sw,
                        change_mean=d.change.mean(), change_sd=d.change.std(), p_change=p_chg,
                        share_of_gap_retained=(d.sliding_window.mean() / d.direct_truncation.mean()
                                               if abs(d.direct_truncation.mean()) > 1e-9 else np.nan)))
le = pd.DataFrame(le_rows)
save_csv(le.round(4), 'length_effect_summary.csv')

pd.set_option('display.width', 220)
print('\nTruncation extent:'); print(pd.DataFrame(extent_rows).round(1).to_string(index=False))
print('\nEquivalence check (worst per task):')
print(pd.concat(eq_all).groupby('task').max_abs_diff.max().map('{:.2e}'.format).to_string())
print('\nReproduction of Table 4 (direct truncation), mean |difference| per task:')
print(rep.assign(a=rep.difference.abs()).groupby('task').a.mean().round(4).to_string())
print('\nSliding window minus direct truncation (QWK):')
print(pd.DataFrame(cmp_rows).round(3).to_string(index=False))
print('\nLength effect under each method:')
print(le.round(3).to_string(index=False))
print(f'\nAll outputs saved to: {RUN_DIR}')
