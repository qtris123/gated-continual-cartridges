from __future__ import annotations
from abc import ABC, abstractmethod
import abc
import asyncio
import os
import random
from typing import Any, List, Optional, Literal, Callable
from pydantic import BaseModel
from pydrantic import ObjectConfig

from cartridges.data.chunkers import Chunker


class Resource(abc.ABC):

    class Config(ObjectConfig):
        _pass_as_config = True
        year: Optional[int] = None
        include_year: bool = False
    
    async def setup(self):
        """This is an optional method that can be used to setup the resource.
        It is called before the first call to sample_prompt.
        """
        pass
    
    @abc.abstractmethod
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        raise NotImplementedError()
    
    
    def to_string(self) -> str:
        raise NotImplementedError("This resource does not implement a string representation.")

SEED_TYPES = Literal[
    "structuring", "summarization", "aggregation", "question", "use_case", "creative", "generic",
    "genconvo_factual", "genconvo_knowledge", "genconvo_disjoint", "genconvo_synthesize",
    "genconvo_structure", "genconvo_creative", "genconvo_counting", "genconvo_reasoning",
]


class TextResource(Resource):
    
    class Config(Resource.Config):
        text: str
        chunker: Chunker.Config

        seed_prompts: List[SEED_TYPES]

    def __init__(self, config: Config):
        self.config = config
        self.text = self.config.text
        self.chunker = None
    
    async def setup(self):
        self.chunker = self.config.chunker.instantiate(text=self.text)
    
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if self.chunker is None:
            raise ValueError("Chunker not initialized. Call setup() first.")
        
        chunk = self.chunker.sample_chunk()
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size, year=self.config.year, include_year=self.config.include_year)
        return chunk, seed_prompts

class TextFileResource(TextResource):

    class Config(Resource.Config):
        path: str
        seed_prompts: List[SEED_TYPES]
        chunker: Chunker.Config

    def __init__(self, config: Config):
        self.config = config
        self.chunker = None
    
    async def setup(self):
        self.text = open(self.config.path).read()
        await super().setup()

class DirectoryResource(Resource):

    class Config(Resource.Config):
        path: str
        included_extensions: List[str] = [".py", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".xml"]
        seed_prompts: List[SEED_TYPES]

    def __init__(self, config: Config):
        self.config = config
        self.files = []

    async def setup(self):
        # Get all files in the directory that match the included extensions
        all_files = [f for f in os.listdir(self.config.path) if os.path.isfile(os.path.join(self.config.path, f))]
        self.files = [
            f for f in all_files 
            if any(f.endswith(ext) for ext in self.config.included_extensions)
        ]

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        if not self.files:
            raise ValueError("No files found in directory. Make sure to call setup() first and check that the directory contains files with the specified extensions.")
        
        # Select a random file
        selected_file = random.choice(self.files)
        file_path = os.path.join(self.config.path, selected_file)
        
        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # If file can't be decoded as UTF-8, try with latin-1 or skip
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception:
                content = f"[Unable to read file {selected_file}]"
        
        # Create context with file information
        context = f"File: {selected_file}\n\n{content}"
        
        # Generate seed prompts
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        
        return context, seed_prompts

class BaseStructuredResource(Resource, ABC):
    """This base class is to be used for resources that can be structured as a nested 
    object containing lists and dictionaries (e.g. JSON objects).
    """

    class Config(Resource.Config):
        seed_prompts: List[SEED_TYPES]
        leaves_only: bool = False
    
    def __init__(self, config: Config):
        self.config = config
        self.data = self._load_data()
        self.ctxs = self._list_nested_data(self.data)
    
    @abc.abstractmethod
    def _load_data(self) -> Any:
        raise NotImplementedError()
    
    def _list_nested_data(self, data: Any, _path: str = "") -> List[(str, str)]:
        """This function creates a string representation of 
        
        Return:
            (path, representation) where path is a string of the path within the object
            to the representation (e.g. "abc/0/def/1") and the string representation of the data. 
        """
        result = []
        
        if isinstance(data, dict):
            # Include the dict itself if not leaves_only
            if not self.config.leaves_only:
                result.append((_path, str(data)))
            
            for key, value in data.items():
                new_path = f"{_path}/{key}" if _path else key
                if isinstance(value, (dict, list)):
                    result.extend(self._list_nested_data(value, new_path))
                else:
                    result.append((new_path, str(value)))
        elif isinstance(data, list):
            # Include the list itself if not leaves_only
            if not self.config.leaves_only:
                result.append((_path, str(data)))
            
            for i, item in enumerate(data):
                new_path = f"{_path}/{i}" if _path else str(i)
                if isinstance(item, (dict, list)):
                    result.extend(self._list_nested_data(item, new_path))
                else:
                    result.append((new_path, str(item)))
        else:
            # For non-dict, non-list data, return the current path and string representation
            result.append((_path, str(data)))
        
        return result
        
    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        path, obj_str = random.choice(self.ctxs)
        ctx = f"The following is located at {path}: {obj_str}"
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

class JSONResource(BaseStructuredResource):
    class Config(BaseStructuredResource.Config):
        path: str
    
    def _load_data(self):
        import json
        return json.load(open(self.config.path))

# --- begin seed prompt generators  ---

def structuring_seed_prompt(**kwargs):
    DATA_FORMATS = [
        "JSON",
        "YAML",
        "TOML",
        "INI",
        "XML",
        "plain text",
    ]

    data_format = random.choice(DATA_FORMATS)

    EXAMPLES = [
        (
            "Can you structure the information in {{subsection}} of {{document}} related to {{something specific}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
        (
            "Can you structure the information in {{subsection}} of {{document}} "
            f"in the following format: {data_format}? "
            "Be sure to include precise information like any dates, times, names, and numerical values.'"
        ),
    ]

    example = random.choice(EXAMPLES)

    return (
        f"Please generate a single chat message instructing an LLM to structure the information in {data_format}. "
        "Output only the chat message itself and absolutely nothing else. "
        "Make sure it is clear what section and document you are asking about. "
        f"The message can follow the following template, filling in details from the corpus: \n\n'{example}'"
    )


def summarization_seed_prompt(**kwargs):
    prompts = [
        (
            "Please generate a single chat message instructing an LLM to summarize part of the corpus. "
            "Make sure the instruction is very explicit about the section of the corpus that you want to summarize. "
            "Include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
        ),
        (
            "Please generate a single chat message instructing an LLM to summarize a section. "
            "Make sure the instruction is explicit about the section that should be summarized and the document it is from."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def question_seed_prompt(**kwargs):
    prompts = [
        (
            "Generate a question for an LLM that will test its knowledge of the information in the corpus above. "
            "In your question be sure to include details (ids, names, titles, dates, etc.) that make it clear what you are asking about. "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "Generate a message for an LLM that will test its knowledge of the information in the corpus above."
            "Be sure to include details (ids, names, titles, dates, etc.) in the question so that it can be answered without access to the corpus (i.e. closed-book setting). "
            "Output only a single question. Do NOT include any other text or explanation other than the question."
        ),
        (
            "You are helping to quiz a user about the information in the corpus. "
            "Please generate a question about the subsection of the corpus above. "
            "Be sure to include details (ids, names, titles, dates, etc.) in the question to make it clear what you are asking about. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    prompt = random.choice(prompts)
    return prompt


def use_case_seed_prompt(**kwargs):
    prompt = (
        "You are working to train a language model on the information in the following corpus. "
        "Your primary goal is to think about practical, real-world tasks or applications that someone could achieve using the knowledge contained within this corpus. "
        "Consider how a user might want to apply this information, not just recall it. "
        "After considering potential use cases, your task will be to generate a sample question that reflects one of these downstream applications. "
        "This question/instruction/task should be something a user, who has access to this corpus, might ask when trying to accomplish their specific goal. "
        "Output only a single question. Do NOT include any other text or explanation other than the question."
    )
    return prompt


def creative_seed_prompt(**kwargs):
    prompt = [
        (
            "You are having a creative conversation inspired by the information in the corpus. "
            "Please generate a question for your conversation partner to start off the discussion. "
            "Answer only with the question, do not include any other text."
        ),
    ]
    return random.choice(prompt)


def generic_seed_prompt(**kwargs):
    return (
        f"Please generate a single chat message to begin a conversation about the information in the corpus. Ask a question about the corpus or make a request."
    )


def genconvo_factual_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question to test someone’s ability to remember factual details from the document. The
        answer should be a few tokens long and be a factual detail from the statement, such as a number, entity,
        date, title, or name.
        This question should not be common knowledge: instead, it should be something that is only answerable
        via information in the document.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year’s\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_knowledge_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question that requires combining information mentioned both inside and outside the
        document.
        This question should require using a fact from the document and also a fact that you are confident about,
        but is not mentioned in the document. For instance: - What are the founding dates of the companies
        that got acquired this year? This is a good question because the names of the acquired companies are
        mentioned in the document and the founding dates are not mentioned. - What is the name of the CEO’s
        spouse? This is a good question because the name of the CEO is mentioned in the document and the
        spouse’s name is not mentioned.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year’s\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_disjoint_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a multi-hop question that tests someone’s ability to use factual information mentioned
        in at least two very different sub-sections of the document.
        This question shouldn’t be a standard question about this kind of document. Instead, it should ask
        about two particularly disconnected ideas, like comparing information about the amount of owned space
        for the company headquarters with the amount of dollars of estimated liability or comparing the revenue
        number with the number of employees.
        This question should also test one’s ability to do retrieval: do not give away part of the answer in
        the question. Ensure that for one to get the correct answer to the question, they need to understand
        the document.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year’s\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_synthesize_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question that requires synthesizing and aggregating information in the document.
        For instance, you could ask someone to summarize a page of the document, list all the key competitors
        mentioned in the document, or summarize the company’s business model
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year’s\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_structure_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question that requires understanding the structure of the document.
        This question should be more about the structure of the document, rather than the precise statement
        details. For instance, you could ask someone to list the titles of all the sections in the document,
        describe the document structure, report the total number of pages, ask which section amongst two sections
        comes first, or report the section with the largest number of tables.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year's\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_creative_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question about the document to test someone’s ability to comprehend the content of the
        document. This question specifically should be focused on their ability to generalize the information
        about the document to a strange question of sorts.
        This question shouldn’t be a standard question about this kind of document, it should ask to do something
        abnormal and creative, like writing a poem about a financial document.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year’s\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_counting_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question that requires counting how frequently different events occur in the document.
        This question should be about statistical properties of the document, rather than the statement details.
        For instance, you could ask someone to count the number of times the word "million" is mentioned or
        count the length of the shortest section title.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year's\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


def genconvo_reasoning_seed_prompt(year=None, include_year=False, **kwargs):
    base_prompt = """
        Please generate a question that requires mathematical reasoning over the values in the document.
        This question should require going beyond the facts directly mentioned in the statement, such as asking
        to compute the percentage increase in revenue between two years, find the largest expense category, or
        calculate difference in profit between two years.
        Do NOT include any other text or explanation other than the question
        """
    if year is not None:
        if include_year:
            return base_prompt + f"\nIf the question involves numerical values, make sure to mention that the data is from the {year} document."
        else:
            return base_prompt + "\nIMPORTANT: If the question involves numerical values, refer to them as being from \"this year's\" document. Do NOT mention any specific year number in the question — use \"this year\" instead."
    return base_prompt


SEED_PROMPT_REGISTRY: dict[SEED_TYPES, Callable] = {
    "structuring": structuring_seed_prompt,
    "summarization": summarization_seed_prompt,
    "question": question_seed_prompt,
    "use_case": use_case_seed_prompt,
    "creative": creative_seed_prompt,
    "generic": generic_seed_prompt,
    "genconvo_factual": genconvo_factual_seed_prompt,
    "genconvo_knowledge": genconvo_knowledge_seed_prompt,
    "genconvo_disjoint": genconvo_disjoint_seed_prompt,
    "genconvo_synthesize": genconvo_synthesize_seed_prompt,
    "genconvo_structure": genconvo_structure_seed_prompt,
    "genconvo_creative": genconvo_creative_seed_prompt,
    "genconvo_counting": genconvo_counting_seed_prompt,
    "genconvo_reasoning": genconvo_reasoning_seed_prompt,
}

def sample_seed_prompts(seed_types: List[SEED_TYPES], batch_size: int, **kwargs) -> List[str]:
    seed_types = random.choices(seed_types, k=batch_size)
    return [SEED_PROMPT_REGISTRY[seed_type](**kwargs) for seed_type in seed_types]

# --- end generators for 