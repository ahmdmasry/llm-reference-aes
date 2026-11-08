"""Run the complete reference-based AES pipeline for OS Question 3.

Set ``AES_DATASET_CSV`` to the independently downloaded question CSV. The file
must contain ``ID``, ``essay``, and ``score`` columns. Outputs default to
``local_outputs/OS3`` inside the package and can be redirected with
``AES_OUTPUT_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline_common import PipelineConfig, run_pipeline


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    run_pipeline(
        PipelineConfig(
            task="OS3",
            score_min=0,
            score_max=5,
            dataset_csv=Path(
                os.environ.get("AES_DATASET_CSV", PACKAGE_ROOT / "data" / "OS3.csv")
            ),
            references_csv=Path(
                os.environ.get(
                    "AES_REFERENCES_CSV",
                    PACKAGE_ROOT
                    / "artifacts"
                    / "generated_references"
                    / "all_generated_references.csv",
                )
            ),
            output_dir=Path(
                os.environ.get(
                    "AES_OUTPUT_DIR", PACKAGE_ROOT / "local_outputs" / "OS3"
                )
            ),
        )
    )
