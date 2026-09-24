#!/usr/bin/env python3
"""Materialize LongHealth 5-phase eval parquets, corpora, and train symlinks.

Splits (sequential patient grouping):
  Phase 1: patient_01, patient_02, patient_03, patient_04 (80 questions)
  Phase 2: patient_05, patient_06, patient_07, patient_08 (80 questions)
  Phase 3: patient_09, patient_10, patient_11, patient_12 (80 questions)
  Phase 4: patient_13, patient_14, patient_15, patient_16 (80 questions)
  Phase 5: patient_17, patient_18, patient_19, patient_20 (80 questions)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent.parent

import sys
sys.path.insert(0, str(ROOT))
from cartridges.data.longhealth.phases import PHASE_TO_PATIENT_IDS

FULL_STRING_TEMPLATE = """\
<patient-record-{patient_id}>
Below is patient {name}'s medical record (ID: {patient_id}). 
They were born on {birthday} and have the following diagnosis: {diagnosis}.
The patients medical record consists of {num_notes} notes included below.
<notes>
{notes}
</notes>
</patient-record-{patient_id}>"""


def setup_train_symlinks() -> None:
    train_dir = ROOT / "data/longhealth/train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for phase in range(1, 6):
        target = ROOT / f"data/longhealth/synth/p{phase:02d}/self_study-n8192/artifact/dataset.parquet"
        link = train_dir / f"qwen_longhealth_p{phase}_task_8192.parquet"
        if not target.exists():
            print(f"Warning: target {target} does not exist yet")
            continue
        rel_target = os.path.relpath(target, train_dir)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(rel_target)
        print(f"  symlink: {link.name} -> {rel_target}")


def build_phase_corpus(patient_ids: list[str], bench_data: dict) -> str:
    out = "Below is a panel of patient records."
    for pid in patient_ids:
        p = bench_data[pid]
        note_keys = sorted(
            p["texts"].keys(),
            key=lambda k: int(re.search(r"\d+", k).group()) if re.search(r"\d+", k) else k,
        )
        notes = "\n".join(f"<{nid}>\n{p['texts'][nid]}\n</{nid}>" for nid in note_keys)
        out += "\n\n"
        out += FULL_STRING_TEMPLATE.format(
            name=p["name"],
            patient_id=pid,
            birthday=p["birthday"],
            diagnosis=p["diagnosis"],
            num_notes=len(p["texts"]),
            notes=notes,
        )
    return out


def build_phase_evals() -> None:
    phases_dir = ROOT / "data/longhealth/phases"
    phases_dir.mkdir(parents=True, exist_ok=True)

    bench_path = ROOT / "data/longhealth/benchmark_v5.json"
    with open(bench_path) as f:
        bench_data = json.load(f)

    # Load source eval parquets
    eval_dir = ROOT / "data/longhealth/eval"
    t1 = pq.read_table(eval_dir / "patients_01_to_10.parquet")
    t2 = pq.read_table(eval_dir / "patients_11_to_20.parquet")

    # Index rows by question_id
    rows_by_qid: dict[str, dict] = {}
    for table in (t1, t2):
        messages_col = table["messages"].to_pylist()
        metadata_col = table["metadata"].to_pylist()
        for msg, meta in zip(messages_col, metadata_col):
            qid = meta["question_id"]
            rows_by_qid[qid] = {
                "messages": msg,
                "metadata": meta,
            }

    print(f"Loaded {len(rows_by_qid)} eval questions from source parquets.")

    manifest: dict[str, dict] = {}
    cumulative_tokens = 0

    for phase in range(1, 6):
        patient_ids = PHASE_TO_PATIENT_IDS[phase]
        phase_rows = []

        for pid in patient_ids:
            p_bench = bench_data[pid]
            patient_info = (
                f"ID {pid}, Name: {p_bench['name']}, "
                f"Birthday: {p_bench['birthday']}, Diagnosis: {p_bench['diagnosis']}"
            )
            for q in p_bench["questions"]:
                qid = f"{pid}_{q['No']}"
                if qid not in rows_by_qid:
                    raise KeyError(f"Question {qid} not found in eval source tables")

                correct_text = q["correct"]
                options = [
                    q["answer_a"],
                    q["answer_b"],
                    q["answer_c"],
                    q["answer_d"],
                    q["answer_e"],
                ]
                option_letters = ["a", "b", "c", "d", "e"]

                # Build Quality-style prompt: question + lettered options, no CoT suffix.
                # The "Answer:" anchor is prepended at decode time by run_accuracy.sh.
                options_block = "\n".join(
                    f"({option_letters[i]}) {opt}" for i, opt in enumerate(options)
                )
                user_content = (
                    f"Please answer the question below about the following patient: {patient_info}"
                    f"\n\n<question>\n{q['question']}\n</question>"
                    f"\n\n<options>\n{options_block}\n</options>"
                    f"\nOutput only the letter of the correct option (e.g. (a), (b), (c), (d), or (e)) along with the content of the option."
                )

                # Ensure category: "longhealth_mcq" and options list are in metadata
                meta = {
                    "patient_id": pid,
                    "question_id": qid,
                    "category": "longhealth_mcq",
                    "options": options,
                    "correct": correct_text,
                }

                phase_rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": correct_text},
                        ],
                        "system_prompt": "",
                        "metadata": meta,
                        "type": "longhealth_mcq",
                    }
                )

        eval_out = phases_dir / f"phase{phase}_eval.parquet"

        schema = pa.schema([
            ("messages", pa.list_(pa.struct([
                ("role", pa.string()),
                ("content", pa.string()),
            ]))),
            ("system_prompt", pa.string()),
            ("metadata", pa.struct([
                ("patient_id", pa.string()),
                ("question_id", pa.string()),
                ("category", pa.string()),
                ("options", pa.list_(pa.string())),
                ("correct", pa.string()),
            ])),
            ("type", pa.string()),
        ])

        table_out = pa.Table.from_pylist(phase_rows, schema=schema)
        pq.write_table(table_out, eval_out)
        print(f"  wrote {eval_out.name}: {len(phase_rows)} questions (patients: {patient_ids})")

        # Materialize phase corpus text
        corpus_text = build_phase_corpus(patient_ids, bench_data)
        txt_out = phases_dir / f"phase{phase}.txt"
        txt_out.write_text(corpus_text)

        # Rough token approximation / length
        n_tokens = len(corpus_text.split())
        cumulative_tokens += n_tokens

        manifest[str(phase)] = {
            "text": str(txt_out.relative_to(ROOT)),
            "eval": str(eval_out.relative_to(ROOT)),
            "tokens": n_tokens,
            "questions": len(phase_rows),
            "cumulative_tokens": cumulative_tokens,
            "patients": patient_ids,
        }

    manifest_path = phases_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  wrote {manifest_path.relative_to(ROOT)}")


def main():
    print("=== 1. Setting up LongHealth train symlinks ===")
    setup_train_symlinks()
    print("\n=== 2. Building 5-phase LongHealth evals and corpora ===")
    build_phase_evals()
    print("\n=== LongHealth pipeline preparation complete ===")


if __name__ == "__main__":
    main()
