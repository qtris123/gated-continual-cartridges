"""Clean a synthesized dataset parquet file for downstream use.

Fixes:
  - Drops metadata.initial_system_prompt (2.4 GB duplicate of system_prompt,
    causes ArrowNotImplementedError with chunked arrays)
  - Casts logprobs from float32 → float64 (double) for precision
  - Casts token_id from int32 → int64

Usage:
    python scripts/clean_parquet.py input.parquet output.parquet
"""

import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def clean_parquet(input_path: str, output_path: str):
    pf = pq.ParquetFile(input_path)

    # Read flat columns + messages (these work fine)
    flat_table = pf.read(columns=["system_prompt", "type"])
    msg_table = pf.read(columns=["messages"])

    # Read metadata sub-fields individually
    # (skip initial_system_prompt — unreadable + duplicate of system_prompt)
    seed_table = pf.read(columns=["metadata.seed_prompt"])
    tc_table = pf.read(columns=["metadata.tool_calls"])

    df = flat_table.to_pandas()
    df["messages"] = msg_table.column("messages").to_pylist()

    seed_list = seed_table.column("metadata").to_pylist()
    tc_list = tc_table.column("metadata").to_pylist()
    df["metadata"] = [
        {"seed_prompt": s["seed_prompt"], "tool_calls": t["tool_calls"]}
        for s, t in zip(seed_list, tc_list)
    ]

    # Rebuild via from_pylist — Python floats become double, ints become int64
    table = pa.Table.from_pylist(df.to_dict(orient="records"))
    pq.write_table(table, output_path, compression="snappy")

    # Verify
    df_check = pd.read_parquet(output_path)
    print(f"Saved to: {output_path}")
    print(f"Shape: {df_check.shape}")
    print(f"Columns: {df_check.columns.tolist()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a synthesized dataset parquet file")
    parser.add_argument("input", help="Path to input dataset.parquet")
    parser.add_argument("output", help="Path to output dataset_clean.parquet")
    args = parser.parse_args()
    clean_parquet(args.input, args.output)
