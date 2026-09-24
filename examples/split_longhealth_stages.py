#!/usr/bin/env python3

"""
Split the two LongHealth datasets into 5 continual-learning stages
and upload them to:

qtris123/gated-continual-cartridges-data

Output structure:

data/longhealth/
└── synth/
    ├── p01/
    │   └── self_study-n8192/
    │       ├── source.json
    │       └── artifact/
    │           └── dataset.parquet
    ├── p02/
    ├── p03/
    ├── p04/
    └── p05/

Stage mapping (sequential patient grouping):

p01: patient_01, patient_02, patient_03, patient_04
p02: patient_05, patient_06, patient_07, patient_08
p03: patient_09, patient_10, patient_11, patient_12
p04: patient_13, patient_14, patient_15, patient_16
p05: patient_17, patient_18, patient_19, patient_20

Requirements:

    pip install -U datasets huggingface_hub pyarrow

Authentication:

    hf auth login

or:

    export HF_TOKEN=hf_...
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


# ---------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------

LOW_SOURCE = "qtris123/llama_0_longhealth-p1-10_8192_no-cartridge"
HIGH_SOURCE = "qtris123/longhealth-p11-20_8192_no-cartridge"

DEST_REPO = "qtris123/gated-continual-cartridges-data"
DEST_ROOT = "data/longhealth"

TECHNIQUE = "self_study-n8192"

EXPECTED_COLUMNS = [
    "messages",
    "system_prompt",
    "metadata",
    "type",
]


# Matches text such as:
#
#     ID: patient_09
#     ID: patient_20
#
PATIENT_RE = re.compile(
    r"\bID:\s*patient_(\d+)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Patient extraction
# ---------------------------------------------------------------------

def extract_patient_id(system_prompt: str) -> int:
    """
    Extract patient number from a LongHealth system prompt.

    Example:

        "ID: patient_09"

    returns:

        9
    """

    match = PATIENT_RE.search(system_prompt or "")

    if match is None:
        raise ValueError(
            "Could not find a patient ID such as "
            "'ID: patient_09' in system_prompt."
        )

    return int(match.group(1))


def add_patient_id(
    ds: Dataset,
    source_name: str,
) -> Dataset:
    """
    Add a temporary `_patient_id` column used for splitting.
    """

    def parse(row):
        return {
            "_patient_id": extract_patient_id(
                row["system_prompt"]
            )
        }

    ds = ds.map(
        parse,
        desc=f"Extracting patient IDs from {source_name}",
    )

    counts = Counter(ds["_patient_id"])

    print()
    print(source_name)
    print(f"  rows: {len(ds):,}")
    print(f"  patient IDs: {sorted(counts)}")
    print(
        "  counts:",
        dict(sorted(counts.items())),
    )

    return ds


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_source(
    ds: Dataset,
    expected_ids: set[int],
    source_name: str,
) -> None:
    """
    Verify that a source dataset contains exactly the expected patients
    and original top-level columns.
    """

    actual_ids = set(ds["_patient_id"])

    if actual_ids != expected_ids:
        raise ValueError(
            f"{source_name} patient IDs do not match expectation.\n"
            f"Expected: {sorted(expected_ids)}\n"
            f"Found:    {sorted(actual_ids)}"
        )

    original_columns = [
        column
        for column in ds.column_names
        if column != "_patient_id"
    ]

    if original_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{source_name} schema changed.\n"
            f"Expected columns: {EXPECTED_COLUMNS}\n"
            f"Found columns:    {original_columns}"
        )


# ---------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------

def select_patients(
    ds: Dataset,
    patient_ids: set[int],
) -> Dataset:
    """
    Keep rows belonging only to the selected patients.
    """

    return ds.filter(
        lambda row: row["_patient_id"] in patient_ids,
        desc=f"Selecting patients {sorted(patient_ids)}",
    )


# ---------------------------------------------------------------------
# Feature compatibility
# ---------------------------------------------------------------------

def make_compatible_for_concat(
    reference: Dataset,
    other: Dataset,
) -> Dataset:
    """
    The two source repos may use slightly different Arrow string
    representations, e.g. string vs large_string.

    Cast the second dataset to the first dataset's feature schema before
    concatenating.
    """

    if other.features != reference.features:
        print(
            "Casting second source to matching feature schema "
            "before concatenation."
        )
        other = other.cast(reference.features)

    return other


# ---------------------------------------------------------------------
# Stage construction
# ---------------------------------------------------------------------

def build_stages(
    low_ds: Dataset,
    high_ds: Dataset,
    output_root: Path,
) -> None:
    """
    Construct five stages with sequential patients:
        p01 -> 1, 2, 3, 4
        p02 -> 5, 6, 7, 8
        p03 -> 9, 10, 11, 12
        p04 -> 13, 14, 15, 16
        p05 -> 17, 18, 19, 20
    """

    print("\nBuilding stages...")

    for stage_idx in range(1, 6):

        # -------------------------------------------------------------
        # Patient IDs for this stage
        # -------------------------------------------------------------

        stage_start = 4 * (stage_idx - 1) + 1
        expected_stage_ids = set(range(stage_start, stage_start + 4))

        low_ids = {pid for pid in expected_stage_ids if pid <= 10}
        high_ids = {pid for pid in expected_stage_ids if pid > 10}

        stage_name = f"p{stage_idx:02d}"

        print()
        print(
            f"{stage_name}: "
            f"patients {sorted(expected_stage_ids)}"
        )

        # -------------------------------------------------------------
        # Filter source datasets
        # -------------------------------------------------------------

        stage_parts = []
        if low_ids:
            low_stage = select_patients(low_ds, low_ids).remove_columns("_patient_id")
            print(f"  low-source rows:  {len(low_stage):,}")
            stage_parts.append(low_stage)
        if high_ids:
            high_stage = select_patients(high_ds, high_ids).remove_columns("_patient_id")
            print(f"  high-source rows: {len(high_stage):,}")
            if stage_parts:
                high_stage = make_compatible_for_concat(stage_parts[0], high_stage)
            stage_parts.append(high_stage)

        if len(stage_parts) == 1:
            stage_ds = stage_parts[0]
        else:
            stage_ds = concatenate_datasets(stage_parts)

        # -------------------------------------------------------------
        # Validate stage contents
        # -------------------------------------------------------------

        written_ids = {
            extract_patient_id(prompt)
            for prompt in stage_ds["system_prompt"]
        }

        if written_ids != expected_stage_ids:
            raise RuntimeError(
                f"{stage_name} validation failed.\n"
                f"Expected patients: "
                f"{sorted(expected_stage_ids)}\n"
                f"Found patients:    "
                f"{sorted(written_ids)}"
            )

        if stage_ds.column_names != EXPECTED_COLUMNS:
            raise RuntimeError(
                f"{stage_name} has incorrect columns.\n"
                f"Expected: {EXPECTED_COLUMNS}\n"
                f"Found:    {stage_ds.column_names}"
            )

        # -------------------------------------------------------------
        # Directory layout
        # -------------------------------------------------------------

        stage_dir = (
            output_root
            / "synth"
            / stage_name
            / TECHNIQUE
        )

        artifact_dir = (
            stage_dir
            / "artifact"
        )

        artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # Write dataset.parquet
        # -------------------------------------------------------------

        parquet_path = (
            artifact_dir
            / "dataset.parquet"
        )

        stage_ds.to_parquet(
            str(parquet_path)
        )

        # -------------------------------------------------------------
        # Write metadata
        # -------------------------------------------------------------

        source_info = {
            "dataset": "longhealth",
            "stage": stage_name,
            "technique": TECHNIQUE,

            "source_datasets": [
                LOW_SOURCE,
                HIGH_SOURCE,
            ],

            "patients": [
                f"patient_{patient_id:02d}"
                for patient_id
                in sorted(expected_stage_ids)
            ],

            "num_rows": len(stage_ds),

            "artifact_path": (
                f"{DEST_ROOT}/"
                f"synth/{stage_name}/"
                f"{TECHNIQUE}/"
                f"artifact/dataset.parquet"
            ),
        }

        source_json_path = (
            stage_dir
            / "source.json"
        )

        with open(
            source_json_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                source_info,
                f,
                indent=2,
            )

            f.write("\n")

        print(
            f"  total rows: {len(stage_ds):,}"
        )

        print(
            f"  wrote: {parquet_path}"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-upload",
        action="store_true",
        help=(
            "Build and validate the staged datasets "
            "without uploading to Hugging Face."
        ),
    )

    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help=(
            "Hugging Face write token. "
            "Defaults to HF_TOKEN or cached HF login."
        ),
    )

    parser.add_argument(
        "--keep-local",
        type=Path,
        default=None,
        help=(
            "Optional directory in which to keep "
            "the generated longhealth tree."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load datasets
    # -----------------------------------------------------------------

    print("Loading source datasets...")

    low_ds = load_dataset(
        LOW_SOURCE,
        split="train",
    )

    high_ds = load_dataset(
        HIGH_SOURCE,
        split="train",
    )

    # -----------------------------------------------------------------
    # Extract patient IDs
    # -----------------------------------------------------------------

    low_ds = add_patient_id(
        low_ds,
        LOW_SOURCE,
    )

    high_ds = add_patient_id(
        high_ds,
        HIGH_SOURCE,
    )

    # -----------------------------------------------------------------
    # Validate source datasets
    # -----------------------------------------------------------------

    validate_source(
        low_ds,
        set(range(1, 11)),
        LOW_SOURCE,
    )

    validate_source(
        high_ds,
        set(range(11, 21)),
        HIGH_SOURCE,
    )

    print("\nSource validation passed.")

    # -----------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------

    if args.keep_local is not None:

        output_root = (
            args.keep_local
            .resolve()
        )

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        cleanup = False

    else:

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="longhealth_stages_"
            )
        )

        output_root = (
            temp_dir
            / "longhealth"
        )

        cleanup = True

    try:

        print(
            f"\nBuilding stages in:\n"
            f"  {output_root}"
        )

        build_stages(
            low_ds,
            high_ds,
            output_root,
        )

        # -------------------------------------------------------------
        # Optional dry run
        # -------------------------------------------------------------

        if args.no_upload:

            print()
            print(
                "--no-upload specified; "
                "nothing was pushed to Hugging Face."
            )

            print(
                f"Generated files:\n"
                f"  {output_root}"
            )

            return

        # -------------------------------------------------------------
        # Upload
        # -------------------------------------------------------------

        print()
        print(
            f"Uploading to:\n"
            f"  {DEST_REPO}"
        )

        api = HfApi(
            token=args.token
        )

        api.upload_folder(
            folder_path=str(output_root),

            path_in_repo=DEST_ROOT,

            repo_id=DEST_REPO,

            repo_type="dataset",

            commit_message=(
                "Add LongHealth 5-stage patient splits "
                "(p01-p05, patients 01-20)"
            ),
        )

        print()
        print("Upload complete.")

        print(
            "https://huggingface.co/datasets/"
            f"{DEST_REPO}/tree/main/"
            f"{DEST_ROOT}/synth"
        )

    finally:

        # Only delete temporary output.
        # --keep-local directories are preserved.

        if cleanup:
            shutil.rmtree(
                output_root.parent,
                ignore_errors=True,
            )


if __name__ == "__main__":
    main()
    