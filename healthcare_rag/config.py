import os
import logging
import openai
from typing import List, Optional, Dict, Any, cast

from weaviate.client import WeaviateAsyncClient
from weaviate.connect import ConnectionParams

from .pipeline.medical_rag import MedicalRAG

logger = logging.getLogger("MedicalRAG")

def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Get an environment variable, with optional default and requirement flag.
    
    Args:
        name: Name of the environment variable
        default: Default value if not found
        required: Whether to raise an error if not found
        
    Returns:
        The value of the environment variable, or the default
        
    Raises:
        ValueError: If required is True and the environment variable is not set
    """
    value = os.environ.get(name, default)
    if value is None:
        if required:
            raise ValueError(f"Required environment variable {name} is not set")
        return ""  # Return empty string for None to fix type issues
    return value

async def setup_medical_rag(
    weaviate_host: Optional[str] = None,
    weaviate_port: Optional[int] = None,
    weaviate_grpc_port: Optional[int] = None,
    collection_names: Optional[List[str]] = None,
    openai_key: Optional[str] = None,
    use_secure_connection: bool = False,
) -> MedicalRAG:
    """
    Set up and return a configured MedicalRAG instance.
    
    Args:
        weaviate_host: Host for Weaviate server (defaults to env var WEAVIATE_HOST or 127.0.0.1)
        weaviate_port: HTTP port for Weaviate (defaults to env var WEAVIATE_PORT or 8080)
        weaviate_grpc_port: gRPC port for Weaviate (defaults to env var WEAVIATE_GRPC_PORT or 50051)
        collection_names: Names of collections to query (defaults to ["Lipitor", "Metformin"])
        openai_key: OpenAI API key (defaults to env var OPENAI_APIKEY)
        use_secure_connection: Whether to use HTTPS/secure gRPC
        
    Returns:
        A configured MedicalRAG instance ready to use
        
    Raises:
        ValueError: If openai_key is not provided and OPENAI_APIKEY env var is not set
    """
    # Define default Weaviate connection parameters from env vars or provided args
    host = weaviate_host or get_env_var("WEAVIATE_HOST", "127.0.0.1")
    http_port_str = weaviate_port or get_env_var("WEAVIATE_PORT", "8080")
    grpc_port_str = weaviate_grpc_port or get_env_var("WEAVIATE_GRPC_PORT", "50051")
    
    # Ensure we have integers for ports
    http_port = int(http_port_str) 
    grpc_port = int(grpc_port_str)

    # Get OpenAI API key
    api_key = openai_key or get_env_var("OPENAI_APIKEY", required=True)

    # Set up headers for OpenAI
    headers = {"X-OpenAI-Api-Key": api_key}

    logger.info(f"Connecting to Weaviate at {host}:{http_port}")

    # Create and connect to Weaviate client
    client = WeaviateAsyncClient(
        connection_params=ConnectionParams.from_params(
            http_host=host,
            http_port=http_port,
            http_secure=use_secure_connection,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=use_secure_connection,
        ),
        additional_headers=headers,
    )

    try:
        # Connect to Weaviate
        await client.connect()
        logger.info("Connected to Weaviate successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise

    # Define collection names to use
    cols = collection_names or ["Lipitor", "Metformin"]

    # Create and return MedicalRAG instance
    return MedicalRAG(
        weaviate_client=client,
        collection_names=cols,
        llm_model="gpt-4o-mini",
    ) 