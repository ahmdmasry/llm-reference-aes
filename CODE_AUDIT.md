# Code audit

The public scripts were reviewed for syntax, sensitive information, obsolete
notebook commands, hard-coded personal paths, duplicate imports, unused imports,
dead code, and cross-validation leakage.

Changes made for the public package include:

- removed Colab notebook magics and automatic Drive mounting;
- removed unused imports, duplicate setup blocks, and task-aggregation code not
  used by the two examples;
- replaced personal storage paths with package-relative paths and environment
  variables;
- consolidated duplicated ASAP and OS logic in `code/pipeline_common.py`;
- fitted standardization inside each hyperparameter-selection fold by using a
  scikit-learn pipeline;
- moved large-model imports into the functions that use them;
- ensured CUDA tensors are moved to CPU before NumPy conversion;
- validated that paired comparisons use identical held-out response IDs;
- standardized use of the packaged, text-redacted reference-answer file; and
- reduced the error-case selector to the reported 26 extreme cases: one
  overprediction and one underprediction for each of 13 tasks/questions.

All scripts pass Python syntax compilation. Lightweight smoke tests covered
reference loading, fold-safe Ridge fitting, paired bootstrapping, and diversity
pair construction. Full feature extraction and truncation analysis require the
independently obtained datasets, model downloads, and substantial compute, so
they were not rerun during this packaging audit.
