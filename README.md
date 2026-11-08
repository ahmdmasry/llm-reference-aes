# LLM-Generated Reference Answers for Similarity-Based AES

This package contains the public reproducibility artifacts for the study of
reference length/detail, generator, and semantic-feature effects in
similarity-based automated essay scoring (AES).

## Contents

- `code/`: one complete ASAP example and one complete OS example, plus the
  revision-analysis scripts.
- `prompts/`: representative Claude prompt templates. The same procedure and
  task-specific substitutions were used for GPT and Gemini.
- `docs/Appendix_Prompt_and_Reference_Samples.docx`: reader-friendly Claude
  prompt examples and selected GPT reference-answer samples.
- `artifacts/generated_references/`: final stored reference answers from GPT,
  Gemini, and Claude. Dataset prompts/source passages have been removed.
- `artifacts/numeric_features/`: numerical features with student-response text
  removed.
- `results/`: primary fold-safe outputs, paper-table data, revision audits, and
  a redacted error-case file.
- `figures/`: figures supporting the achieved-length and truncation analyses.
- `MANIFEST.csv`: file sizes and SHA-256 checksums.

## Data are not redistributed

The ASAP and OS datasets are not included. Download them from the original
public sources listed in `DATA_ACCESS.md`, then update the input paths in the
two sample scripts.

## Suggested execution order

1. Install the dependencies in `requirements.txt`.
2. Download the relevant dataset and place it in a local data directory.
3. Place the task CSV in `data/ASAP1.csv` or `data/OS3.csv`, or set the
   `AES_DATASET_CSV` environment variable to its location.
4. Run `code/sample_ASAP1_pipeline.py` or `code/sample_OS3_pipeline.py`.
5. Run the scripts in `code/revision_analyses/` as needed.

The sample pipelines use the generated references already included in this
package. Outputs are written under `local_outputs/` by default. Set
`AES_OUTPUT_DIR` to use another location.

The code contains no API keys, passwords, email addresses, account identifiers,
or hard-coded personal storage paths. It does not call any commercial LLM API;
the LLMs were used only to produce the stored reference-answer artifacts.

The bundled result files are the study outputs. The public scripts were later
cleaned and corrected as documented in `CODE_AUDIT.md`; the full GPU analyses
were not rerun during packaging. Therefore, exact result parity should be
verified before describing the cleaned scripts as the archival execution code.

## Reference-generation protocol

Reference answers were generated through the standard user interfaces of GPT
5.1, Claude Sonnet 4.5, and Gemini 3. Interface defaults were used where model
settings such as temperature and top-p were not exposed. The stored final
reference answers are provided. Regeneration attempts were not systematically
logged, so the package does not report a reliable regeneration count.

## License and citation

The Python code is released under the MIT License. Generated references,
derived features, results, figures, prompts, and supplementary documents are
released under CC BY 4.0. See `LICENSES.md` for the exact scope.

Citation metadata are provided in `CITATION.cff`. The repository URL is
<https://github.com/ahmdmasry-uaeu/llm_reference_aes>. Add the release DOI after
the first public release is archived.
