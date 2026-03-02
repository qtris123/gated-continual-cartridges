"""Local inference client that runs a FlexLlama model with an optional trained cartridge.

No Tokasaurus server needed — the model and cartridge are loaded directly into GPU memory.

Supports:
  - No cartridge (plain model)
  - Local .pt cartridge file  (cartridge_path)
  - HuggingFace-hosted cartridge  (cartridge_hf_id)
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.clients.base import Client, ClientConfig, ClientResponse, ClientSample
from cartridges.clients.usage import Usage
from cartridges.generation import flex_generate
from cartridges.utils import get_logger

logger = get_logger(__name__)


class LocalCacheClient(Client):
    """Runs FlexLlamaForCausalLM locally, optionally conditioned on a trained cartridge."""

    class Config(ClientConfig):
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
        cartridge_path: Optional[str] = None   # local .pt file
        cartridge_hf_id: Optional[str] = None  # HuggingFace repo ID
        device: str = "cuda"
        dtype: str = "bfloat16"

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def __init__(self, config: "LocalCacheClient.Config"):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._cache: Optional[TrainableCache] = None
        self._generate_lock = threading.Lock()

    def _setup(self):
        """Lazy initialisation: load model, tokenizer, and optional cartridge."""
        if self._model is not None:
            return

        from cartridges.models import FlexLlamaForCausalLM

        dtype = getattr(torch, self.config.dtype)
        device = self.config.device

        logger.info(f"Loading model: {self.config.model_name}")
        self._model = (
            FlexLlamaForCausalLM.from_pretrained(self.config.model_name)
            .to(device)
            .to(dtype)
        )
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Resolve cartridge path (local or HuggingFace)
        cartridge_path = self.config.cartridge_path
        if cartridge_path is None and self.config.cartridge_hf_id is not None:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading cartridge from HuggingFace: {self.config.cartridge_hf_id}")
            repo_dir = snapshot_download(repo_id=self.config.cartridge_hf_id)
            # Look for a .pt file inside the downloaded snapshot
            import os
            pt_files = [f for f in os.listdir(repo_dir) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(
                    f"No .pt file found in HuggingFace repo {self.config.cartridge_hf_id} "
                    f"(downloaded to {repo_dir})"
                )
            cartridge_path = os.path.join(repo_dir, pt_files[0])
            logger.info(f"Using cartridge file: {cartridge_path}")

        if cartridge_path is not None:
            logger.info(f"Loading cartridge: {cartridge_path}")
            self._cache = TrainableCache.from_pretrained(cartridge_path, device=device)
            self._cache = self._cache.to(device).to(dtype)
            n_total = self._cache.num_cartridge_tokens()
            logger.info(
                f"Cartridge loaded: {self._cache._num_frozen_tokens} frozen + "
                f"{self._cache._num_trainable_tokens} trainable = {n_total} tokens"
            )
            print(
                f"[LocalCacheClient] Cartridge token count: "
                f"{self._cache._num_frozen_tokens} frozen + "
                f"{self._cache._num_trainable_tokens} trainable = {n_total} total"
            )

    # -------------------------------------------------------------------------
    # chat()
    # -------------------------------------------------------------------------

    async def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        max_completion_tokens: int,
        temperature: float = 0.6,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ClientResponse:
        """Generate responses for a batch of chats using the local model."""
        self._setup()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            self._generate_sync,
            chats,
            max_completion_tokens,
            temperature,
        )
        return response

    # -------------------------------------------------------------------------
    # Synchronous generation (called via run_in_executor)
    # -------------------------------------------------------------------------

    def _generate_sync(
        self,
        chats: List[List[Dict[str, Any]]],
        max_new_tokens: int,
        temperature: float,
    ) -> ClientResponse:
        with self._generate_lock:
            device = self.config.device
            dtype = getattr(torch, self.config.dtype)

            # --- Build flat (input_ids, seq_ids, position_ids) tensors ---
            all_input_ids: List[torch.Tensor] = []
            all_seq_ids: List[torch.Tensor] = []
            all_position_ids: List[torch.Tensor] = []

            for idx, chat in enumerate(chats):
                ids = self._tokenizer.apply_chat_template(
                    chat,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)  # shape: (1, seq_len)
                flat = ids.flatten()
                n = flat.shape[0]
                all_input_ids.append(flat)
                all_seq_ids.append(torch.full((n,), idx, dtype=torch.long, device=device))
                all_position_ids.append(torch.arange(n, device=device))

            input_ids = torch.cat(all_input_ids, dim=0)
            seq_ids = torch.cat(all_seq_ids, dim=0)
            position_ids = torch.cat(all_position_ids, dim=0)

            # --- Generate ---
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                generated: Dict[int, List[int]] = flex_generate(
                    model=self._model,
                    tokenizer=self._tokenizer,
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    cache=self._cache,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

            # --- Build ClientResponse ---
            samples: List[ClientSample] = []
            total_completion_tokens = 0
            total_prompt_tokens = input_ids.shape[0]

            for idx in range(len(chats)):
                token_ids = generated.get(idx, [])
                text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
                samples.append(ClientSample(text=text, token_ids=token_ids))
                total_completion_tokens += len(token_ids)

            usage = Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
            )
            return ClientResponse(samples=samples, usage=usage)
