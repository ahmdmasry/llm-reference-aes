# Version 1.0.0

This is the first public reproducibility release for the study
"LLM-Generated Reference Answers for Similarity-Based Automated Essay Scoring."

## Included

- Sample ASAP Task 1 and OS Question 3 pipelines
- GPT, Gemini, and Claude generated reference-answer artifacts
- Student-text-redacted numerical features
- Fold-safe scoring results and statistical outputs
- Reference-length and tokenizer audits
- Sliding-window sensitivity analysis materials
- Reference-diversity analysis materials
- Redacted 26-case qualitative error-analysis materials
- Representative prompts, figures, and supplementary appendix

## Not included

- ASAP or OS datasets
- Raw student responses
- Full qualitative-review workbooks containing response text
- Credentials, private storage paths, or exported account information

## Reproduction note

The bundled results are the archived study outputs. The public scripts were
subsequently cleaned and corrected as described in `CODE_AUDIT.md`; the full
GPU analyses were not rerun during packaging. Users should verify exact result
parity before treating the cleaned scripts as the archival execution code.
