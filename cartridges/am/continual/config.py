"""The Phase-2 run config and the stage bundle it instantiates.

``AMContinualConfig`` is a ``RunConfig`` in its own right: Phase 2 is a
closed-form, backprop-free write, so it never needed ``TrainConfig``'s
optimizer, epochs, or batch-size fields, and dressing it up as one made
``execution_mode`` necessary to dispatch away from the train loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydrantic import RunConfig

from cartridges.am.components.beta import BetaFitter
from cartridges.am.components.keys import KeyWriter
from cartridges.am.components.objective import ValueObjective
from cartridges.am.components.queries import ReferenceQueries
from cartridges.am.components.slots import SlotSelector
from cartridges.am.components.teacher import TeacherTarget
from cartridges.cache import KVCacheFactory
from cartridges.models.config import ModelConfig
from cartridges.train import LossEvalConfig


@dataclass(frozen=True)
class AMStages:
    """The six stage objects plus the run-level constants they share.

    Built once per run by ``AMContinualConfig.build_stages`` and then read-only:
    every stage is stateless, so the document loop stays the only thing that
    carries state from one document to the next.
    """

    slots: SlotSelector
    queries: ReferenceQueries
    teacher: TeacherTarget
    keys: KeyWriter
    beta: BetaFitter
    objective: ValueObjective

    rope_theta: float
    compute_stats: bool


class AMContinualConfig(RunConfig):
    """One closed-form AM write per document, over a cartridge from Phase 1.

    Field-by-field reference: ``research_loop/AM_CONFIG.md``. Ablation numbers
    and verdicts: ``research_loop/state/mechanism_registry.md``.
    """

    name: str = "am_continual"
    # Runtime output root; set explicitly by the endpoint (continual_write) rather
    # than read from the environment at import time (recipes-as-config).
    output_dir: str = "."

    model: ModelConfig
    kv_cache_initializer: KVCacheFactory.Config  # the Phase-1 cartridge
    document_data_path: str  # the MT parquet, grouped into documents

    # MECH-003: rotary base for the `doc_rope_offset = T_doc` rotation between
    # reference queries and document keys. Top-level, NOT per stage: the teacher
    # target, the key rewrite and the oracle write all consume it, and composing
    # two rotations only lands on an absolute position when both use the same
    # theta. The default is the historical hard-coded value, NOT the model's own
    # base (Qwen3-4B-Instruct-2507 = 5e6).
    rope_theta: float = 10000.0

    slots: SlotSelector.Config = Field(default_factory=SlotSelector.Config)
    queries: ReferenceQueries.Config = Field(default_factory=ReferenceQueries.Config)
    teacher: TeacherTarget.Config = Field(default_factory=TeacherTarget.Config)
    keys: KeyWriter.Config = Field(default_factory=KeyWriter.Config)
    beta: BetaFitter.Config = Field(default_factory=BetaFitter.Config)
    objective: ValueObjective.Config = Field(default_factory=ValueObjective.Config)

    loss_evals: list[LossEvalConfig] = Field(default_factory=list)

    save_after_each_document: bool = True
    compute_update_stats: bool = True

    # Read by `cartridges.train.save_cache` / `evaluate_perplexity`, which this
    # run shares with the gradient path.
    keep_last_n_saved: int = 1
    save_to_wandb: bool = False
    # Results go to disk only (`config.yaml`, `phase2_summary.json`,
    # `per_document_am_stats.pt`, `am_doc_*.pt`). Declared so the shared helpers
    # can read it; typed `None` so it cannot be switched on by accident.
    wandb: None = None

    device: str = "cuda"
    seed: int = 42

    def build_stages(self) -> AMStages:
        """Instantiate the six stages once, for the whole run."""
        return AMStages(
            slots=self.slots.instantiate(),
            queries=self.queries.instantiate(),
            teacher=self.teacher.instantiate(),
            keys=self.keys.instantiate(),
            beta=self.beta.instantiate(),
            objective=self.objective.instantiate(),
            rope_theta=self.rope_theta,
            compute_stats=self.compute_update_stats,
        )

    def run(self):
        # Local import: `run` imports this module for the config type, so a
        # module-scope import here would be circular.
        from cartridges.am.continual.run import run_am_continual

        return run_am_continual(self)
