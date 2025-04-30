import logging
import argparse
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
    TesseractOcrOptions,
)
from docling.chunking import HybridChunker # type: ignore
from transformers import AutoTokenizer # type: ignore
import json
import os
from pathlib import Path

_log = logging.getLogger(__name__)


class DocumentChunkProcessor:
    """
    Convert a PDF to a docling document, chunk it, merge table fragments,
    and stitch back any sentences accidentally split under the same heading.
    """

    def __init__(
        self,
        source: str = "docs/metformin.pdf",
        embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 8192,
        num_threads: int = 4,
    ) -> None:
        """
        Args:
            source: PDF path or URI. Defaults to docs/00021412.pdf.
            embed_model_id: HF embed model for tokenization.
            max_tokens: max tokens per chunk.
            num_threads: CPU threads for OCR.
        """
        self.source = source
        # tokenizer & chunker
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer, max_tokens=max_tokens, merge_peers=True # type: ignore
        )
        self.doc_name = source.split("/")[-1].split(".")[0]

        # PDF→docling converter options
        pdf_opts = PdfPipelineOptions()
        pdf_opts.do_ocr = False
        pdf_opts.do_table_structure = True
        pdf_opts.table_structure_options.do_cell_matching = True
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.AUTO
        )
        pdf_opts.generate_parsed_pages = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            }
        )

    def convert_document(self):
        """Run PDF → docling document conversion."""
        return self.converter.convert(source=self.source).document

    def chunk_document(self, document):
        """Split the docling document into elementary chunks."""
        return list(self.chunker.chunk(dl_doc=document))

    def _merge_table_chunks(self, chunks):
        """Merge all chunks belonging to the same table (by meta.doc_items[].self_ref)."""
        table_groups = defaultdict(list)
        others = []

        for idx, chunk in enumerate(chunks):
            ref = None
            if hasattr(chunk, "meta") and getattr(chunk.meta, "doc_items", None):
                for item in chunk.meta.doc_items:
                    if (
                        hasattr(item, "self_ref")
                        and isinstance(item.self_ref, str)
                        and item.self_ref.startswith("#/tables/")
                    ):
                        ref = item.self_ref
                        table_groups[ref].append((idx, chunk))
                        break
            if ref is None:
                others.append((idx, chunk))

        merged = []
        for ref, group in table_groups.items():
            group.sort(key=lambda x: x[0])
            merged_text = "\n".join(c.text for _, c in group)
            first_meta = group[0][1].meta
            merged_chunk = type(group[0][1])(text=merged_text, meta=first_meta)
            merged.append((group[0][0], merged_chunk))

        merged.extend(others)
        merged.sort(key=lambda x: x[0])
        return [c for _, c in merged]

    def _should_merge_split(self, c1, c2):
        """
        Heuristic: same non-empty headings list AND
                   c1.text doesn't end in '.' AND
                   c2.text starts lowercase α.
        """
        h1 = getattr(c1.meta, "headings", None)
        h2 = getattr(c2.meta, "headings", None)
        if not (isinstance(h1, list) and h1 and h1 == h2):
            return False

        t1, t2 = c1.text.strip(), c2.text.strip()
        if not t1 or not t2 or t1.endswith("."):
            return False

        return t2[0].isalpha() and t2[0].islower()

    def _merge_split_sentences(self, chunks):
        """Merge sentence-fragments split under the same heading."""
        result = []
        skip = False

        for i in range(len(chunks)):
            if skip:
                skip = False
                continue

            if i + 1 < len(chunks) and self._should_merge_split(chunks[i], chunks[i + 1]):
                combined = chunks[i].text + " " + chunks[i + 1].text
                new_chunk = type(chunks[i])(text=combined, meta=chunks[i].meta)
                result.append(new_chunk)
                skip = True
            else:
                result.append(chunks[i])

        return result

    def process(self):
        """Full pipeline: convert, chunk, merge tables & split-sentence fragments."""
        doc = self.convert_document()
        chunks = self.chunk_document(doc)
        chunks = self._merge_table_chunks(chunks)
        return self._merge_split_sentences(chunks)

    def print_chunks(self, chunks):
        """Log each chunk's text, token-count, context, and metadata."""
        for idx, chunk in enumerate(chunks):
            count = len(self.tokenizer.tokenize(chunk.text))
            _log.info(f"=== Chunk {idx} ({count} tokens) ===\n{chunk.text!r}")
            ctx = self.chunker.contextualize(chunk=chunk)
            ctx_count = len(self.tokenizer.tokenize(ctx))
            _log.info(f"--- Contextualized ({ctx_count} tokens) ---\n{ctx}")
            _log.info(f"--- Meta ---\n{chunk.meta}\n")

    def save_chunks(self, chunks, output_dir="data"):
        """Save all chunks' text, contextualized text, and metadata to a single json file."""
        # Ensure data directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a list of all chunks
        all_chunks = []
        for idx, chunk in enumerate(chunks):
            ctx = self.chunker.contextualize(chunk=chunk)
            all_chunks.append({
                "id_": idx,  # Changed to id_ to match EXPECTED_PROPERTIES in vector_store
                "text": chunk.text,
                "contextualized": ctx,
                "metadata": chunk.meta.export_json_dict(),
                "page_numbers": get_page_numbers(chunk),
                "doc_source": self.doc_name
            })
        
        # Save all chunks to a single file
        output_path = Path(output_dir) / f"chunks_{self.doc_name}.json"
        with open(output_path, "w") as f:
            f.write(json.dumps(all_chunks, indent=4))
        
        return output_path

def get_page_numbers(chunk):
    """Get the page numbers of the chunk."""
    return [prov.page_no for prov in chunk.meta.doc_items[0].prov]

def run_chunker(source_path, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                max_tokens=8192, output_dir="data"):
    """Helper function to run the chunker from other modules."""
    proc = DocumentChunkProcessor(
        source=source_path,
        embed_model_id=model_name,
        max_tokens=max_tokens,
    )
    chunks = proc.process()
    output_path = proc.save_chunks(chunks, output_dir=output_dir)
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description="Convert & chunk a PDF, merging tables and split sentences."
    )
    parser.add_argument(
        "--source", 
        default="docs/metformin.pdf", 
        help="Path or URI to the PDF file (default: docs/metformin.pdf)"
    )
    parser.add_argument(
        "--model", 
        default="sentence-transformers/all-MiniLM-L6-v2", 
        help="HF embed model"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=8192, 
        help="Max tokens per chunk"
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save the output chunks JSON (default: data)"
    )

    args = parser.parse_args()

    proc = DocumentChunkProcessor(
        source=args.source,
        embed_model_id=args.model,
        max_tokens=args.max_tokens,
    )
    final = proc.process()
    proc.print_chunks(final)
    proc.save_chunks(final, output_dir=args.output_dir)

if __name__ == "__main__":
    main() 