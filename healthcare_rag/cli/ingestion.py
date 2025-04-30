"""
CLI commands for data ingestion and vector database management.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple

from healthcare_rag.processors.pdf_chunker import run_chunker
from healthcare_rag.storage.vector_store import ingest_json_to_collection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf(args: argparse.Namespace) -> Optional[str]:
    """
    Process a PDF file to generate chunks.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Path to the generated JSON file or None if processing failed.
    """
    try:
        logger.info(f"Processing PDF file: {args.source}")
        output_path = run_chunker(
            source_path=args.source,
            model_name=args.model,
            max_tokens=args.max_tokens,
            output_dir=args.output_dir
        )
        logger.info(f"Chunks saved to: {output_path}")
        return str(output_path)
    except Exception as e:
        logger.exception(f"Error processing PDF: {e}")
        return None


def load_to_weaviate(args: argparse.Namespace) -> bool:
    """
    Load JSON data into Weaviate.
    
    Args:
        args: Command line arguments.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info(f"Loading data from {args.json_file} to collection {args.collection}")
        success = ingest_json_to_collection(
            collection_name=args.collection,
            json_file_path=args.json_file,
            delete_existing=args.delete_existing,
            batch_size=args.batch_size
        )
        if success:
            logger.info("Data loaded successfully")
        else:
            logger.error("Failed to load data")
        return success
    except Exception as e:
        logger.exception(f"Error loading data to Weaviate: {e}")
        return False


def run_pipeline(args: argparse.Namespace) -> Tuple[bool, Optional[str], Optional[bool]]:
    """
    Run the full pipeline: PDF processing and Weaviate ingestion.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Tuple containing (pipeline_success, json_path, ingestion_success)
    """
    # Process the PDF
    json_path = process_pdf(args)
    if not json_path:
        logger.error("PDF processing failed, pipeline aborted")
        return False, None, None
    
    # Load to Weaviate if requested
    ingestion_success = None
    if args.ingest:
        collection_name = args.collection or Path(args.source).stem
        ingest_args = argparse.Namespace(
            collection=collection_name,
            json_file=json_path,
            delete_existing=args.delete_existing,
            batch_size=args.batch_size
        )
        ingestion_success = load_to_weaviate(ingest_args)
    
    return True, json_path, ingestion_success


def main():
    """Main CLI entry point for data ingestion commands."""
    parser = argparse.ArgumentParser(
        description="Healthcare RAG Data Ingestion Tools"
    )
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Ingestion command to run"
    )
    
    # PDF processing command
    pdf_parser = subparsers.add_parser(
        "process-pdf", 
        help="Process a PDF document into chunks"
    )
    pdf_parser.add_argument(
        "source", 
        help="Path to the PDF file"
    )
    pdf_parser.add_argument(
        "--model", 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model for tokenization"
    )
    pdf_parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=8192,
        help="Maximum tokens per chunk"
    )
    pdf_parser.add_argument(
        "--output-dir", 
        default="data",
        help="Directory to save the chunks JSON"
    )
    
    # Weaviate ingestion command
    weaviate_parser = subparsers.add_parser(
        "load-weaviate", 
        help="Load JSON data into Weaviate"
    )
    weaviate_parser.add_argument(
        "collection", 
        help="Weaviate collection name"
    )
    weaviate_parser.add_argument(
        "json_file", 
        help="Path to the JSON file with chunks"
    )
    weaviate_parser.add_argument(
        "--delete-existing", 
        action="store_true",
        help="Delete the collection if it exists"
    )
    weaviate_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Batch size for data import"
    )
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline", 
        help="Run the full ingestion pipeline: PDF → Chunks → Weaviate"
    )
    pipeline_parser.add_argument(
        "source", 
        help="Path to the PDF file"
    )
    pipeline_parser.add_argument(
        "--model", 
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model for tokenization"
    )
    pipeline_parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=8192,
        help="Maximum tokens per chunk"
    )
    pipeline_parser.add_argument(
        "--output-dir", 
        default="data",
        help="Directory to save the chunks JSON"
    )
    pipeline_parser.add_argument(
        "--ingest", 
        action="store_true",
        help="Load the chunks into Weaviate after processing"
    )
    pipeline_parser.add_argument(
        "--collection", 
        help="Weaviate collection name (defaults to PDF filename)"
    )
    pipeline_parser.add_argument(
        "--delete-existing", 
        action="store_true",
        help="Delete the collection if it exists"
    )
    pipeline_parser.add_argument(
        "--batch-size", 
        type=int, 
        default=100,
        help="Batch size for data import"
    )
    
    args = parser.parse_args()
    
    if args.command == "process-pdf":
        process_pdf(args)
    elif args.command == "load-weaviate":
        load_to_weaviate(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 