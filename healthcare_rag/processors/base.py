import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml
import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..services.llm import LLMParserService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MedicalRAG")

# Type definitions
ResponseModel = TypeVar("ResponseModel", bound="BaseModel")

# Custom timing decorator for instrumentation
def log_timing(func):
    """Decorator to log the time taken for a function to execute."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return wrapper


class PromptManager:
    """
    Loads Jinja templates from a directory, renders them with context,
    and returns a list of OpenAI chat message dicts.
    """

    def __init__(self, templates_dir: str | Path = "prompts"):
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(enabled_extensions=("j2",)),
        )
        # Cache compiled templates
        self._cache = {}

    def _get_template(self, name: str):
        if name not in self._cache:
            template_path = f"{name}.yaml.j2"
            self._cache[name] = self.env.get_template(template_path)
        return self._cache[name]

    def messages(self, name: str, **context) -> List[ChatCompletionMessageParam]:
        """
        Render a template file and return a list of messages.
        """
        raw = self._get_template(name).render(**context)
        return yaml.safe_load(raw)


class BaseProcessor:
    """Base class for all LLM-based processors to reduce code duplication."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        async_client: Optional[openai.AsyncOpenAI] = None,
        prompt_manager: Optional[PromptManager] = None,
        parser_service: Optional[LLMParserService] = None,
    ):
        self.llm_model = llm_model
        self.async_client = async_client
        self.pm = prompt_manager or PromptManager()
        self.parser_service = parser_service

    async def _call_llm(
        self,
        prompt_name: str,
        temperature: float = 0.1,
        response_format: Type[ResponseModel] = Any,  # type: ignore
        default_response: Optional[ResponseModel] = None,
        **prompt_args,
    ) -> Optional[ResponseModel]:
        """Standardized method for LLM calls using prompt templates."""
        messages = self.pm.messages(prompt_name, **prompt_args)
        if self.parser_service is None:
            logger.error("LLM parser service is not initialized")
            return default_response
        return await self.parser_service.parse_completion(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            default_response=default_response,
        )

    async def _call_llm_completions(
        self,
        prompt_name: str,
        temperature: float = 0.1,
        default_response: str = "",
        **prompt_args,
    ) -> str:
        """Standardized method for LLM streaming using prompt templates."""
        messages = self.pm.messages(prompt_name, **prompt_args)
        if self.async_client is None:
            logger.error("OpenAI async client is not initialized")
            return default_response
        response = await self.async_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
        )
        if response.choices[0].message.content is None:
            return default_response
        return response.choices[0].message.content 