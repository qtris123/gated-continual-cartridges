import os
import time
from typing import Any, Dict, List, Literal, Optional, Type
import openai
from openai.types.chat.chat_completion import ChatCompletion
import asyncio
import tiktoken
import numpy as np
from pydrantic import BaseConfig
from cartridges.clients.base import Client, ClientSample, ClientConfig, ClientResponse, TopLogprobs
from cartridges.clients.usage import Usage, num_tokens_from_messages_flexible
from cartridges.utils import get_logger

_TOKEN_ID_PREFIX = "token_id:"


def _parse_server_token_id(token: str) -> Optional[int]:
    if not token.startswith(_TOKEN_ID_PREFIX):
        return None
    try:
        return int(token[len(_TOKEN_ID_PREFIX):])
    except ValueError:
        return None


class OpenAIClient(Client):
    """This client works with any inference server that supports the OpenAI API.
    It is simply a wrapper around the OpenAI Python client that handles retrying and
    exposes a batch interface with async parallel execution.

    Features:
    - Async batch processing with asyncio.gather for parallel requests
    - Flexible tokenizer support: automatically uses tiktoken for OpenAI models 
      and falls back to Huggingface AutoTokenizer for other models (e.g., Qwen)
    - Message truncation and retry logic for context length issues
    - Conversation tracking for prompt caching
    - Works with any OpenAI-compatible API endpoint

    Example:
        config = OpenAIClient.Config(
            model_name="Qwen/Qwen2.5-7B-Instruct",  # HF model
            base_url="https://your-endpoint.modal.run/v1"
        )
        client = OpenAIClient(config)
        
        response = await client.chat(
            chats=[
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "How are you?"}]
            ],
            temperature=0.7,
            max_completion_tokens=100
        )

    Note: although Tokasaurus also supports the OpenAI API, we have a separate 
    `TokasaurusClient` you should use for self-study (and synthetic data generation). 
    There are two issues with the standard OpenAI API: (1) for very large batch sizes,
    we bottleneck the server with so many concurrent requests -- the batch endpoint is
    not a great option because it requires two calls (one create and one retrieve) that
    need to hit the same replica, which is annoying to orchestrate with e.g. modal 
    autoscaling. (2) the openai api stores logprobs very inefficiently -- it returns a 
    separate object for each logprob, which is very slow to parse.
    Our TokasaurusClient uses a custom batch endpoint that packs logprobs into the 
    the response more efficiently.
    """

    client_class: Type[openai.AsyncOpenAI] = openai.AsyncOpenAI



    class Config(ClientConfig):
        """Configuration options for the OpenAIClient."""
        _pass_as_config: bool = True

        model_name: str = "gpt-4o"
        base_url: Optional[str] = None
        api_key: Optional[str] = None

        # if we max out the context length, retry truncating the messages to fit in the 
        # max length
        truncate_messages_and_retry: bool = True  

        # Set this when the server is launched with `--return-tokens-as-token-ids`
        # (vLLM). The server then reports every token as the string
        # "token_id:{id}", which is the only way to recover true vocabulary ids over
        # the OpenAI API -- it otherwise returns token text, and re-tokenizing text
        # is not guaranteed to invert the model's own segmentation. Without this,
        # top_logprobs token ids are filled with -1 and are unusable as
        # distillation targets.
        return_tokens_as_token_ids: bool = False


    def __init__(self, config: Config):
        """
        Initialize the OpenAI client with the provided config.
        If config.api_key is set, it will override OPENAI_API_KEY in the environment.
        """
        self.config = config

        self.type: Literal["openai", "hf"] = "openai" if config.base_url is None else "hf"
        
        self.client = self.client_class(
            api_key=config.api_key if config.api_key else os.getenv("OPENAI_API_KEY"),
            base_url=config.base_url,
        )
        self.logger = get_logger("OpenAIClient")

        self.conversations = {}

        # Initialize tokenizer - try tiktoken first, then fall back to Huggingface
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize tokenizer - try tiktoken first, then fall back to Huggingface."""
        
        # First, try to use tiktoken for OpenAI models
        try:
            encoding = tiktoken.encoding_for_model(self.config.model_name)
            self.logger.info(f"Using tiktoken encoding for model {self.config.model_name}")
            return encoding
        except KeyError:
            self.logger.info(f"Model {self.config.model_name} not found in tiktoken, trying Huggingface tokenizer")
            pass
        
        # Fall back to Huggingface tokenizer
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.logger.info(f"Using Huggingface tokenizer for model {self.config.model_name}")
            return tokenizer
            
        except Exception as e:
            self.logger.warning(f"Failed to load Huggingface tokenizer for {self.config.model_name}: {e}")
            
            # Final fallback: create a dummy tokenizer that does character-based estimation
            class CharBasedTokenizer:
                def __init__(self, model_name):
                    self.model_name = model_name
                
                def encode(self, text, add_special_tokens=True):
                    # Very rough estimation: 4 characters per token
                    return list(range(len(text) // 4 + 1))
            
            self.logger.warning(f"Using character-based token estimation for {self.config.model_name}")
            return CharBasedTokenizer(self.config.model_name)

    def _encode(self, text: str) -> Optional[List[int]]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            # tiktoken encodings do not take add_special_tokens
            return self.tokenizer.encode(text)
        except Exception as e:
            self.logger.warning(f"Failed to encode text for token ids: {e}")
            return None

    def _extract_with_token_ids(
        self, content: List[Any]
    ) -> tuple[Optional[List[int]], Optional[TopLogprobs]]:
        """Recover true vocabulary ids from a server run with
        `--return-tokens-as-token-ids`, and pack the top-k alternatives into the dense
        [num_tokens, k] layout that `TopLogprobs.flatten` expects."""

        token_ids = []
        for entry in content:
            token_id = _parse_server_token_id(entry.token)
            if token_id is None:
                self.logger.warning(
                    f"return_tokens_as_token_ids is set but the server returned "
                    f"{entry.token!r}, which is not of the form 'token_id:{{id}}'. "
                    "Launch the server with --return-tokens-as-token-ids."
                )
                token_ids = None
                break
            token_ids.append(token_id)

        rows_ids, rows_logprobs = [], []
        for entry in content:
            pairs = []
            for alt in entry.top_logprobs or []:
                alt_id = _parse_server_token_id(alt.token)
                if alt_id is not None:
                    pairs.append((alt_id, alt.logprob))
            # flatten() reads cumulative mass left to right, so rows must be descending
            pairs.sort(key=lambda pair: pair[1], reverse=True)
            rows_ids.append([p[0] for p in pairs])
            rows_logprobs.append([p[1] for p in pairs])

        k = max((len(row) for row in rows_ids), default=0)
        if k == 0:
            return token_ids, None

        # Padding matches FlatTopLogprobs.reconstruct: absent ids are -1, absent
        # logprobs are -1000 so they contribute no probability mass.
        ids_matrix = np.full((len(rows_ids), k), -1, dtype=np.int32)
        logprobs_matrix = np.full((len(rows_logprobs), k), -1000.0, dtype=np.float32)
        for row, (ids, logprobs) in enumerate(zip(rows_ids, rows_logprobs)):
            ids_matrix[row, : len(ids)] = ids
            logprobs_matrix[row, : len(logprobs)] = logprobs

        return token_ids, TopLogprobs(logprobs=logprobs_matrix, token_ids=ids_matrix)

    def _extract_without_token_ids(self, content: List[Any]) -> Optional[TopLogprobs]:
        """Logprobs only, for servers that report token text. The ids are filled with
        -1, so this is not usable for distillation."""

        logprobs_list = []
        for token in content:
            token_logprobs = [token.logprob]
            if token.top_logprobs:
                token_logprobs.extend([t.logprob for t in token.top_logprobs])
            logprobs_list.append(token_logprobs)

        if not logprobs_list:
            return None

        max_len = max(len(row) for row in logprobs_list)
        padded_logprobs = [
            row + [-1000.0] * (max_len - len(row)) for row in logprobs_list
        ]
        return TopLogprobs(
            logprobs=np.array(padded_logprobs, dtype=np.float32),
            token_ids=np.full((len(padded_logprobs), max_len), -1, dtype=np.int32),
        )

    async def chat(
        self,
        chats: List[List[Dict[str, Any]]],
        temperature: float = 0.6,
        stop: List[str] = [],
        max_completion_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        conversation_id: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        cartridges: Optional[List[Dict[str, Any]]] = None,
        modal_upstream_id: Optional[str] = None,
    ) -> ClientResponse:
        assert len(chats) > 0
        
        # Handle legacy single chat format
        if isinstance(chats[0], Dict):
            chats = [chats]

        extra_body = {}
        kwargs = {}
        if enable_thinking is not None and self.type == "hf":
            extra_body = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        elif self.type == "openai":
            kwargs["reasoning_effort"] = "high" if enable_thinking else "low"
        
        if cartridges is not None:
            extra_body["cartridges"] = cartridges
        
        if modal_upstream_id is not None and self.type != "openai":
            extra_body["modal_upstream_id"] = modal_upstream_id
        
        # Create individual async tasks for each chat
        async def process_single_chat(messages: List[Dict[str, Any]]) -> tuple[ChatCompletion, List[Dict[str, Any]]]:
            async def chat_single(m: List[Dict[str, Any]]) -> ChatCompletion:
                return await self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=m,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop if stop else None,
                    n=1,
                    top_logprobs=top_logprobs,
                    logprobs=top_logprobs is not None,
                    extra_body=extra_body,
                    **kwargs,
                )
            
            # Handle message truncation with retry logic
            error = None
            for num_truncated_messages in range(len(messages)):
                try:
                    used_messages = messages[num_truncated_messages:]
                    response: ChatCompletion = await chat_single(used_messages)
                    return response, used_messages
                except openai.BadRequestError as e:
                    error = e
                    if e.body.get("code", "") not in ("context_length_exceeded", "too_many_messages"):    
                        raise e
                    
                    if not self.config.truncate_messages_and_retry:
                        raise ValueError(
                            f"OpenAI returned the following BadRequestError: {e}." 
                            "Set truncate_messages_and_retry=True to retry with truncated messages."
                        )
                    
                    self.logger.warning(
                        f"OpenAI returned the following BadRequestError: {e}. "
                        f"Truncating first {num_truncated_messages + 1} messages and retrying..."
                    )                        
            else:
                raise ValueError(
                    f"OpenAI returned the following BadRequestError: {error}. "
                    "Even though you have set truncate_messages_and_retry=True, "
                    "the last message is still too long."
                )
        
        # Execute all chats in parallel using asyncio.gather
        chat_results = await asyncio.gather(*[process_single_chat(messages) for messages in chats])

        # Process results
        responses = []
        usage = Usage(prompt_tokens=0, completion_tokens=0)
        
        for response, used_messages in chat_results:
            # Handle usage counting
            curr_usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_prompt_tokens=response.usage.prompt_tokens_details.cached_tokens if response.usage.prompt_tokens_details else 0,
            )
            
            # Handle conversation tracking for caching
            if conversation_id is not None:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []

                max_matched_messages = []
                for prev_messages in self.conversations[conversation_id]:
                    matched_messages = []
                    for curr, prev in zip(used_messages, prev_messages):
                        if (curr["content"] != prev["content"]) or (curr["role"] != prev["role"]):
                            break
                        matched_messages.append(curr)
                    if len(matched_messages) > len(max_matched_messages):
                        max_matched_messages = matched_messages

                curr_usage.seen_prompt_tokens = num_tokens_from_messages_flexible(max_matched_messages, tokenizer=self.tokenizer)        
                
                # Add to conversation history
                choice = response.choices[0]
                self.conversations[conversation_id].append(
                    used_messages + [{"role": "assistant", "content": choice.message.content}]
                )
            
            usage += curr_usage
            
            # Process the response choice
            choice = response.choices[0]
            
            # Extract token IDs if available from logprobs
            token_ids = None
            sample_top_logprobs = None
            
            if choice.logprobs and choice.logprobs.content:
                if self.config.return_tokens_as_token_ids:
                    token_ids, sample_top_logprobs = self._extract_with_token_ids(
                        choice.logprobs.content
                    )
                else:
                    sample_top_logprobs = self._extract_without_token_ids(
                        choice.logprobs.content
                    )

            if token_ids is None and self.config.return_tokens_as_token_ids:
                # Logprobs were not requested for this call, so the server told us
                # nothing about ids. Re-encoding the text is what the training dataset
                # falls back to for messages that carry no token ids.
                token_ids = self._encode(choice.message.content)
            
            responses.append(ClientSample(
                text=choice.message.content,
                token_ids=token_ids,
                top_logprobs=sample_top_logprobs,
            ))
        return ClientResponse(samples=responses, usage=usage)

