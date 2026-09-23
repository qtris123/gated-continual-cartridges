"""LongHealth patients grouped into five continual-learning phases.

LongHealth is a clinical benchmark: 20 synthetic patients, each with a full
record of 2-13 notes and exactly 20 multiple-choice questions. A document is one
patient's record; a phase is four patients.

Four per phase comes from the review comment on the plan doc — the doc first
proposed one patient per phase, and one patient is only ~11k tokens, which
defeats the purpose of compaction. Four lands each phase at ~46k tokens, and
because there are exactly 20 patients, five phases of four use every patient
once with nothing left over and no selection judgement to make. Patients are
assigned in ID order.

Phase sizes (Qwen3-4B tokenizer):

    phase 1  patient_01-04  30 notes  48,807 tokens  80 questions
    phase 2  patient_05-08  27 notes  41,977 tokens  80 questions
    phase 3  patient_09-12  24 notes  47,288 tokens  80 questions
    phase 4  patient_13-16  23 notes  46,221 tokens  80 questions
    phase 5  patient_17-20  29 notes  48,748 tokens  80 questions
    total                  133 notes 233,041 tokens 400 questions

Question count is identical across phases, which none of the other four datasets
manage, so LongHealth is the cleanest place to read forgetting curves off.

The diagnoses are heterogeneous within every phase (phase 1 alone spans DLBCL,
melanoma, multiple myeloma and pancreatic cancer), so the phases are not
separated by clinical domain — they are separated by *patient identity*. That is
the intended structure: the question is whether the cartridge still knows
patient_02's record after training through patient_20's.
"""

from __future__ import annotations

from typing import Dict, List

from cartridges.data.longhealth.utils import (
    LongHealthPatient,
    LongHealthQuestion,
    load_longhealth_dataset,
)

NUM_PHASES = 5
PATIENTS_PER_PHASE = 4

PHASE_TO_PATIENT_IDS: Dict[int, List[str]] = {
    1: ["patient_01", "patient_02", "patient_03", "patient_04"],
    2: ["patient_05", "patient_06", "patient_07", "patient_08"],
    3: ["patient_09", "patient_10", "patient_11", "patient_12"],
    4: ["patient_13", "patient_14", "patient_15", "patient_16"],
    5: ["patient_17", "patient_18", "patient_19", "patient_20"],
}


def patients_for_phase(phase: int) -> List[LongHealthPatient]:
    """Load the four patients belonging to ``phase``, in declared order."""
    if phase not in PHASE_TO_PATIENT_IDS:
        raise ValueError(
            f"phase must be one of {sorted(PHASE_TO_PATIENT_IDS)}, got {phase}"
        )
    wanted = PHASE_TO_PATIENT_IDS[phase]
    patients = {p.patient_id: p for p in load_longhealth_dataset(wanted)}
    missing = [pid for pid in wanted if pid not in patients]
    if missing:
        raise ValueError(f"patients missing from the LongHealth release: {missing}")
    return [patients[pid] for pid in wanted]


def questions_for_phase(phase: int) -> List[LongHealthQuestion]:
    """Every question belonging to ``phase``'s patients (20 each, 80 total)."""
    return [q for patient in patients_for_phase(phase) for q in patient.questions]


def build_phases(
    num_phases: int = NUM_PHASES, patients_per_phase: int = PATIENTS_PER_PHASE
) -> Dict[int, List[str]]:
    """Recompute ``PHASE_TO_PATIENT_IDS`` from the dataset, in patient-ID order."""
    patient_ids = sorted(p.patient_id for p in load_longhealth_dataset())
    needed = num_phases * patients_per_phase
    if len(patient_ids) < needed:
        raise ValueError(
            f"LongHealth has {len(patient_ids)} patients, need {needed}"
        )
    return {
        p + 1: patient_ids[p * patients_per_phase : (p + 1) * patients_per_phase]
        for p in range(num_phases)
    }
