from typing import Dict, List, Optional
import json
import os
from pathlib import Path

from pydantic import BaseModel
import requests


class LongHealthAnswerLocation(BaseModel):
    start: List[float]
    end: List[float]


class LongHealthQuestion(BaseModel):
    question_id: str
    question: str
    correct: str

    answer_a: str
    answer_b: str
    answer_c: str
    answer_d: str
    answer_e: str

    answer_location: Optional[Dict[str, LongHealthAnswerLocation]] = None


class LongHealthPatient(BaseModel):
    patient_id: str
    texts: Dict[str, str]
    name: str
    birthday: str
    diagnosis: str
    questions: List[LongHealthQuestion]


DATASET_URL = "https://raw.githubusercontent.com/kbressem/LongHealth/refs/heads/main/data/benchmark_v5.json"


def _cache_path() -> Path:
    root = os.environ.get("CARTRIDGES_DIR", os.getcwd())
    return Path(root) / "data" / "longhealth" / "benchmark_v5.json"


def _load_raw() -> dict:
    path = _cache_path()
    if path.exists():
        with open(path) as f:
            return json.load(f)
    response = requests.get(DATASET_URL, timeout=120)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text)
    return json.loads(response.text)


def load_longhealth_dataset(patient_ids: Optional[List[str]] = None) -> List[LongHealthPatient]:
    data = _load_raw()
    for patient_id, row in data.items():
        for question in row["questions"]:
            question["question_id"] = patient_id + "_" + str(question["No"])

    patients = [
        LongHealthPatient(patient_id=patient_id, **row)
        for patient_id, row in data.items()
        if patient_ids is None or patient_id in patient_ids
    ]
    return patients
