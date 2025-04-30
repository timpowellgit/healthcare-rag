# zzscratches/weaviate_query.py

"""
Script to query a Weaviate collection using nearText search (Asynchronous).

This script connects to a Weaviate instance (using connection logic
from the ingestion script) and performs a semantic search on a specified
collection using the asynchronous client.

Example Usage:
python zzscratches/weaviate_query.py Jeopardy "animals in movies" --limit 2
"""

import argparse
import logging
import sys
import os
import asyncio # Added asyncio
from typing import Optional, Callable, Awaitable, Dict

import weaviate
from weaviate.classes.query import MetadataQuery
# Import the async client and connection params
from weaviate.connect import ConnectionParams
from weaviate.client import WeaviateAsyncClient
# Define connection defaults
DEFAULT_WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
DEFAULT_WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
DEFAULT_WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Placeholder for the async connection function and constants
# Type hint updated for async client
connect_to_weaviate_async: Optional[Callable[[str, int, int], Awaitable[WeaviateAsyncClient]]] = None
WEAVIATE_HOST: str = DEFAULT_WEAVIATE_HOST
WEAVIATE_PORT: int = DEFAULT_WEAVIATE_PORT
WEAVIATE_GRPC_PORT: int = DEFAULT_WEAVIATE_GRPC_PORT
openai_key = os.getenv("OPENAI_APIKEY")
if not openai_key:
    logging.error("OPENAI_APIKEY environment variable not set.")
    raise ValueError("OPENAI_APIKEY environment variable not set.")

headers: Dict[str, str] = {"X-OpenAI-Api-Key": openai_key}

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Default async connection function if import fails
async def default_connect_async(host: str, port: int, grpc_port: int) -> Optional[WeaviateAsyncClient]:
    """Default async connection function."""
    logging.info(f"Attempting default async connection to Weaviate at {host}:{port} (gRPC: {grpc_port})")
    client = WeaviateAsyncClient(
         connection_params=ConnectionParams.from_params(
            http_host=host,
            http_port=port,
            http_secure=False, # Assuming non-secure HTTP for local default
            grpc_host=host, # Assuming gRPC host is the same
            grpc_port=grpc_port,
            grpc_secure=False, # Assuming non-secure gRPC for local default
            # headers=headers # Removed from here
        ),
        additional_headers=headers, # Pass headers here using 'additional_headers'
        # startup_period=5 # Optional: Adjust startup period if needed
    )
    # Try connecting with retry logic
    retries = 3
    for attempt in range(retries):
        try:
            await client.connect() # Connect using configured params
            logging.info("Default async connection successful.")
            return client
        except Exception as e:
            logging.warning(f"Connection attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2) # Wait before retrying
            else:
                logging.error("Failed to connect after multiple retries.")
                raise # Re-raise the last exception
    return None # Return None if connection fails
# Make the function async
async def perform_near_text_query(
    client: WeaviateAsyncClient, # Updated type hint
    collection_name: str,
    query_text: str,
    limit: int
):
    """
    Performs a nearText query on the specified collection asynchronously.

    Args:
        client: The Weaviate async client instance.
        collection_name: The name of the collection to query.
        query_text: The text to search for semantically.
        limit: The maximum number of results to return.
    """
    try:
        # Check if the collection exists (async)
        if not await client.collections.exists(collection_name): # Use await
            logging.error(f"Collection '{collection_name}' does not exist.")
            return None # Return None explicitly on error/missing collection

        collection = client.collections.get(collection_name)
        logging.info(f"Querying collection '{collection_name}' for text similar to: '{query_text}'")

        # Perform query asynchronously
        response = await collection.query.hybrid( # Use await
            query=query_text,
            # Consider making query_properties an argument if needed
            query_properties=["contextualized"],
            limit=limit,
            return_metadata=MetadataQuery(score=True) # Request distance metadata
        )

        logging.info(f"Found {len(response.objects)} results:")
        if not response.objects:
            logging.warning("Query returned no results.")
        return response

    except Exception as e:
        logging.exception(f"An error occurred during the async query for collection '{collection_name}'. Error: {e}")
        return None # Return None on exception


# Make main async
async def main():
    """Main asynchronous execution function."""
    client: Optional[WeaviateAsyncClient] = None # Updated type hint

    connector = connect_to_weaviate_async or default_connect_async

    try:
        # Connect to Weaviate asynchronously
        logging.info("Connecting to Weaviate...")
        client = await connector(WEAVIATE_HOST, WEAVIATE_PORT, WEAVIATE_GRPC_PORT) # Use await

        if not client or not client.is_connected():
             logging.critical("Failed to establish Weaviate connection. Exiting.")
             sys.exit(1)

        # Perform the query
        response = await perform_near_text_query( # Use await
            client,
            "Lipitor", # Still hardcoded collection
            "animals in movies", # Still hardcoded query
            10 # Still hardcoded limit
        )

        # Check if the query returned results before processing
        if response and response.objects:
            for obj in response.objects:
                # Print all properties
                print(obj.properties) # Assuming properties is a dict-like object
                # Print distance if available
                if obj.metadata:
                    print(f"Distance: {obj.metadata.distance}")
                print()
        else:
            # This handles None response from perform_near_text_query or empty results
            logging.warning("Query returned no results or an error occurred during query execution.")

        logging.info("Async query script finished.")

    except Exception as e:
        # Catch potential connection errors from connector() as well
        logging.critical(f"An unexpected error occurred during main async execution: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
    finally:
        # Close the client connection asynchronously if it was successfully opened
        if client and client.is_connected(): # Check connection before closing
            await client.close() # Use await
            logging.info("Weaviate async client connection closed.")

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())