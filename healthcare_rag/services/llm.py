import logging
from typing import List, Optional, Type, TypeVar
from pydantic import BaseModel
import openai
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger("MedicalRAG")

# Type definitions
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)

class LLMParserService:
    """Handles making calls to and parsing responses from OpenAI chat completions."""

    def __init__(self, async_client: "openai.AsyncOpenAI"):
        self.async_client = async_client

    async def parse_completion(
        self,
        *,  # Force keyword arguments
        model: str,
        messages: List[ChatCompletionMessageParam],
        response_format: Type[ResponseModel],
        temperature: float,
        default_response: Optional[ResponseModel] = None,
    ) -> Optional[ResponseModel]:
        """
        Calls the OpenAI API using the beta parse helper and handles errors.

        Args:
            model: The model name to use.
            messages: The list of messages for the prompt.
            response_format: The Pydantic model class for parsing the response.
            temperature: The sampling temperature.
            default_response: The default value to return on failure or if parsing yields None.

        Returns:
            The parsed Pydantic model instance or the default_response.
        """
        format_name = getattr(response_format, "__name__", str(response_format))

        try:
            # Use fully qualified name for logging if it's a Pydantic model
            logger.debug(
                f"Calling LLM (model={model}, temp={temperature}, response_format={format_name})"
            )
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": response_format,
            }
            if model == "o3-mini":
                # remove temperature
                params.pop("temperature")

            # The .parse() method directly returns the parsed Pydantic model or None
            response = await self.async_client.beta.chat.completions.parse(**params)
            parsed_response = response.choices[0].message.parsed

            if parsed_response is None:
                logger.warning(
                    f"LLM response parsing returned None for {format_name}. Returning default."
                )
                return default_response

            return parsed_response
        except Exception as e:
            logger.error(f"Error during LLM call ({format_name}): {e}", exc_info=True)
            return default_response 