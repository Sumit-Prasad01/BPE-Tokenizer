"""Streaming data loader from Hugging Face datasets with exact byte-quota controllers."""

from pathlib import Path
from typing import Generator
import datasets
from tqdm import tqdm

from src.config import SourceConfig, PreprocessingConfig
from src.data.preprocessor import TextPreprocessor
from utils.custom_exception import CorpusDownloadError
from utils.helpers import format_bytes, ensure_dir
from utils.logger import logger


class DatasetStreamDownloader:
    """Streams text documents from Hugging Face datasets until a target byte quota is met."""

    def __init__(self, preprocessing_config: PreprocessingConfig | None = None):
        self.preprocessor = TextPreprocessor(preprocessing_config)

    def stream_source_documents(
        self,
        source: SourceConfig,
        max_bytes: int | None = None,
        progress_bar: bool = True,
    ) -> Generator[str, None, None]:
        """Streams cleaned documents from a single Hugging Face source.
        
        Args:
            source: SourceConfig specifying dataset name, subset, split, text column.
            max_bytes: Target byte threshold. If None, uses source.target_bytes.
            progress_bar: Whether to display a tqdm progress bar.
            
        Yields:
            Cleaned document strings.
        """
        quota_bytes = max_bytes if max_bytes is not None else source.target_bytes
        logger.info(
            f"📥 Initiating stream for '{source.hf_dataset}' "
            f"(subset: {source.subset or 'default'}, split: {source.split}) "
            f"targeting {format_bytes(quota_bytes)}..."
        )

        try:
            # Stream directly without downloading multi-gigabyte archives
            dataset = datasets.load_dataset(
                source.hf_dataset,
                name=source.subset,
                split=source.split,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            error_msg = f"Failed to connect to Hugging Face dataset '{source.hf_dataset}': {e}"
            logger.error(error_msg)
            raise CorpusDownloadError(error_msg) from e

        accumulated_bytes = 0
        doc_count = 0

        pbar = None
        if progress_bar:
            pbar = tqdm(
                total=quota_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Streaming {source.hf_dataset.split('/')[-1]}",
            )

        try:
            for item in dataset:
                raw_text = item.get(source.text_column)
                if not raw_text or not isinstance(raw_text, str):
                    continue

                cleaned_doc = self.preprocessor.clean_text(raw_text)
                if not self.preprocessor.is_valid_document(cleaned_doc):
                    continue

                doc_bytes = len(cleaned_doc.encode("utf-8"))
                accumulated_bytes += doc_bytes
                doc_count += 1

                if pbar:
                    pbar.update(doc_bytes)

                yield cleaned_doc

                if accumulated_bytes >= quota_bytes:
                    logger.info(
                        f"🎯 Target quota achieved for '{source.hf_dataset}': "
                        f"{format_bytes(accumulated_bytes)} across {doc_count:,} documents."
                    )
                    break
        finally:
            if pbar:
                pbar.close()

    def download_source_to_file(
        self,
        source: SourceConfig,
        output_file: str | Path,
        max_bytes: int | None = None,
    ) -> int:
        """Streams documents from a source and writes them directly to a file separated by double newlines.
        
        Returns:
            Total bytes written.
        """
        out_path = Path(output_file)
        ensure_dir(out_path.parent)

        total_bytes = 0
        doc_count = 0

        with open(out_path, "w", encoding="utf-8") as f:
            for doc in self.stream_source_documents(source, max_bytes=max_bytes):
                entry = doc + "\n\n"
                f.write(entry)
                total_bytes += len(entry.encode("utf-8"))
                doc_count += 1

        logger.info(
            f"💾 Successfully saved {format_bytes(total_bytes)} ({doc_count:,} docs) "
            f"to {out_path.resolve()}"
        )
        return total_bytes
