"""
Module to connect to Weaviate, create collections, and import data from JSON files.

This module provides functions to:
- Connect to a local Weaviate instance.
- Create Weaviate collections with a predefined schema.
- Load data from JSON files.
- Import data into specified collections in batches.

Required Environment Variables:
- OPENAI_APIKEY: Your OpenAI API key for the text2vec-openai vectorizer.
"""

import weaviate
from weaviate.classes.config import Property, DataType, Configure
import os
import dotenv
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# --- Configuration ---
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Weaviate Connection Details (can be overridden by environment variables)
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

# Import Settings
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_ERRORS = 10

# Collection Schema Definition
# Define properties expected in the Weaviate schema and potentially the input JSON
# Adjust this list based on your actual schema needs.
EXPECTED_PROPERTIES = {
    "id_": DataType.INT,
    "text": DataType.TEXT,
    "contextualized": DataType.TEXT,
    "doc_source": DataType.TEXT,
    "page_numbers": DataType.INT_ARRAY,
}


def connect_to_weaviate(
    host: str = WEAVIATE_HOST, 
    port: int = WEAVIATE_PORT, 
    grpc_port: int = WEAVIATE_GRPC_PORT
) -> weaviate.WeaviateClient:
    """
    Connects to a Weaviate instance using environment variables for API keys.

    Args:
        host: The hostname or IP address of the Weaviate instance.
        port: The REST API port of the Weaviate instance.
        grpc_port: The gRPC port of the Weaviate instance.

    Returns:
        An initialized WeaviateClient instance.

    Raises:
        ValueError: If the OPENAI_APIKEY environment variable is not set.
        weaviate.exceptions.WeaviateStartUpError: If the connection fails.
    """
    openai_key = os.getenv("OPENAI_APIKEY")
    if not openai_key:
        logging.error("OPENAI_APIKEY environment variable not set.")
        raise ValueError("OPENAI_APIKEY environment variable not set.")

    headers: Dict[str, str] = {"X-OpenAI-Api-Key": openai_key}

    try:
        client = weaviate.connect_to_local(
            skip_init_checks=False,  # Perform checks for a more robust connection attempt
            host=host,
            port=port,
            grpc_port=grpc_port,
            headers=headers,
        )
        if client.is_ready():
            logging.info(f"Weaviate connection established to {host}:{port}")
            return client
        else:
            logging.error(f"Weaviate connection failed: Client not ready.")
            raise weaviate.exceptions.WeaviateStartUpError(
                "Client not ready after connection attempt."
            )
    except Exception as e:
        logging.exception(f"Failed to connect to Weaviate at {host}:{port}. Error: {e}")
        raise


def create_collection(
    client: weaviate.WeaviateClient, collection_name: str
) -> weaviate.Collection:
    """
    Creates a Weaviate collection with a predefined schema.

    Args:
        client: The Weaviate client instance.
        collection_name: The name for the new collection.

    Returns:
        The created Weaviate Collection object.

    Raises:
        weaviate.exceptions.WeaviateQueryError: If collection creation fails.
    """
    try:
        # Check if collection already exists
        if client.collections.exists(collection_name):
            logging.warning(
                f"Collection '{collection_name}' already exists. Skipping creation."
            )
            return client.collections.get(collection_name)

        properties = [
            Property(name=prop_name, data_type=prop_type)
            for prop_name, prop_type in EXPECTED_PROPERTIES.items()
        ]

        client.collections.create(
            collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(),
            properties=properties,
            reranker_config=Configure.Reranker.cohere(),  # Ensure you have COHERE_APIKEY env var if using this
        )
        logging.info(f"Collection '{collection_name}' created successfully.")
        return client.collections.get(collection_name)
    except Exception as e:
        logging.exception(
            f"Failed to create collection '{collection_name}'. Error: {e}"
        )
        raise


def prepare_data_for_import(data_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares a single data row for Weaviate import by keeping only expected properties.

    Args:
        data_row: The raw data dictionary from the JSON file.

    Returns:
        A dictionary containing only the properties defined in EXPECTED_PROPERTIES.
    """
    return {
        key: data_row.get(key)
        for key in EXPECTED_PROPERTIES
        if key in data_row  # Only include keys present in the data_row
    }


def import_data(
    collection: weaviate.Collection,
    data_rows: List[Dict[str, Any]],
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """
    Imports data into the specified Weaviate collection using batching.

    Args:
        collection: The Weaviate Collection object to import into.
        data_rows: A list of dictionaries, where each dictionary represents an object.
        batch_size: The number of objects to include in each batch.
    """
    object_count = len(data_rows)
    logging.info(
        f"Starting data import for {object_count} objects into collection '{collection.name}' with batch size {batch_size}."
    )

    imported_count = 0
    failed_imports = []

    try:
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for i, raw_data_row in enumerate(data_rows):
                properties_to_import = prepare_data_for_import(raw_data_row)

                # Optional: Add validation here to ensure required fields are present
                if not properties_to_import.get("text"):  # Example validation
                    logging.warning(
                        f"Skipping row {i+1} due to missing 'text' property: {raw_data_row.get('id_', 'Unknown ID')}"
                    )
                    continue

                batch.add_object(properties=properties_to_import)

                if batch.number_errors > MAX_BATCH_ERRORS:
                    logging.error(
                        f"Stopping batch import for collection '{collection.name}' due to exceeding {MAX_BATCH_ERRORS} errors."
                    )
                    break

            # Check final batch results after the loop
            if batch.number_errors > 0:
                logging.warning(
                    f"Batch import for '{collection.name}' completed with {len(batch.failed_objects)} failed objects."
                )
                failed_imports.extend(
                    batch.failed_objects
                )  # Capture errors from the last batch

        # Note: collection.batch.failed_objects might not capture all failures if the loop breaks early.
        # The batch context manager handles final error reporting better.
        successful_imports = object_count - len(
            failed_imports
        )  # Approximation if loop broke early

        if failed_imports:
            logging.error(
                f"Total failed imports for '{collection.name}': {len(failed_imports)}"
            )
            # Log details of the first few failures for debugging
            for i, failed in enumerate(failed_imports[:5]):
                logging.error(
                    f"  Failed object #{i+1}: {failed.message} | Original data hint: {failed.original_object}"
                )
        else:
            logging.info(
                f"Successfully imported approximately {successful_imports}/{object_count} objects into '{collection.name}'."
            )

    except Exception as e:
        logging.exception(
            f"An unexpected error occurred during batch import for collection '{collection.name}'. Error: {e}"
        )


def load_data_from_json(json_filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads data from a JSON file.

    Args:
        json_filepath: The Path object pointing to the JSON file.

    Returns:
        A list of dictionaries loaded from the JSON file, or an empty list if loading fails.
    """
    if not json_filepath.is_file():
        logging.error(f"JSON data file not found: {json_filepath}")
        return []
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                logging.error(f"JSON data in {json_filepath} is not a list.")
                return []
            logging.info(
                f"Successfully loaded {len(data)} records from {json_filepath}"
            )
            return data
    except json.JSONDecodeError:
        logging.exception(f"Failed to decode JSON from file: {json_filepath}")
        return []
    except IOError as e:
        logging.exception(f"Failed to read file: {json_filepath}. Error: {e}")
        return []


def ingest_json_to_collection(
    collection_name: str,
    json_file_path: str,
    delete_existing: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    host: str = WEAVIATE_HOST,
    port: int = WEAVIATE_PORT,
    grpc_port: int = WEAVIATE_GRPC_PORT,
) -> bool:
    """
    High-level function to ingest JSON data into a Weaviate collection.
    
    Args:
        collection_name: The name of the Weaviate collection.
        json_file_path: Path to the JSON file containing the data.
        delete_existing: Whether to delete the collection if it exists.
        batch_size: Size of import batches.
        host: Weaviate host.
        port: Weaviate port.
        grpc_port: Weaviate gRPC port.
    
    Returns:
        True if successful, False otherwise.
    """
    client = None
    try:
        # Connect to Weaviate
        client = connect_to_weaviate(host, port, grpc_port)
        
        # Optionally delete collection
        if delete_existing and client.collections.exists(collection_name):
            logging.info(f"Deleting existing collection '{collection_name}'")
            client.collections.delete(collection_name)
        
        # Create or get collection
        collection = create_collection(client, collection_name)
        
        # Load and import data
        json_filepath = Path(json_file_path)
        data = load_data_from_json(json_filepath)
        
        if data:
            import_data(collection, data, batch_size)
            logging.info(f"Completed import to collection '{collection_name}'")
            return True
        else:
            logging.error(f"No data loaded from {json_file_path}. Import failed.")
            return False
            
    except Exception as e:
        logging.exception(f"Failed to ingest data: {str(e)}")
        return False
    finally:
        if client:
            client.close()


def main():
    """Main execution function for CLI usage."""
    parser = argparse.ArgumentParser(description="Weaviate Data Loader")
    parser.add_argument(
        "--collection",
        action="append",
        nargs=2,
        metavar=("COLLECTION_NAME", "JSON_FILE_PATH"),
        help="Specify a collection name and the path to its JSON data file. Can be used multiple times.",
        dest="collections_to_load",
        default=[],
    )
    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete all existing Weaviate collections before creating new ones.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for data import (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=WEAVIATE_HOST,
        help=f"Weaviate host (default: {WEAVIATE_HOST}, or WEAVIATE_HOST env var).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=WEAVIATE_PORT,
        help=f"Weaviate port (default: {WEAVIATE_PORT}, or WEAVIATE_PORT env var).",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=WEAVIATE_GRPC_PORT,
        help=f"Weaviate gRPC port (default: {WEAVIATE_GRPC_PORT}, or WEAVIATE_GRPC_PORT env var).",
    )
    
    args = parser.parse_args()

    client = None
    try:
        # Connect to Weaviate
        client = connect_to_weaviate(args.host, args.port, args.grpc_port)

        # Optionally delete all existing collections
        if args.delete_all:
            logging.warning("Deleting all existing Weaviate collections as requested.")
            try:
                client.collections.delete_all()  # type: ignore
                logging.info("All collections deleted successfully.")
            except Exception as e:
                logging.exception("Failed to delete all collections.")

        # Process each specified collection and data file
        if not args.collections_to_load:
            logging.warning(
                "No collections specified for loading. Use the --collection argument."
            )
            return

        for collection_name, json_file_str in args.collections_to_load:
            logging.info(
                f"Processing collection '{collection_name}' with data from '{json_file_str}'..."
            )
            json_filepath = Path(json_file_str)

            # Create collection
            collection = create_collection(client, collection_name)

            # Load data
            data_to_import = load_data_from_json(json_filepath)

            # Import data
            if data_to_import:
                import_data(collection, data_to_import, args.batch_size)
            else:
                logging.warning(
                    f"No data loaded from {json_filepath} for collection '{collection_name}'. Skipping import."
                )

        logging.info("Data ingestion completed successfully.")

    except Exception as e:
        logging.critical(
            f"An unexpected error occurred during main execution: {e}", exc_info=True
        )

    finally:
        # Close the client connection if it was successfully opened
        if client:
            client.close()
            logging.info("Weaviate client connection closed.")


if __name__ == "__main__":
    main() 