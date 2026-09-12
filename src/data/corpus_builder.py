"""Multi-source corpus assembler that builds balanced training and held-out evaluation datasets."""

import random
from pathlib import Path
from typing import Any

from src.config import DatasetMixConfig
from src.data.downloader import DatasetStreamDownloader
from utils.custom_exception import CorpusBuildError
from utils.helpers import (
    ensure_dir,
    format_bytes,
    format_number,
    compute_file_sha256,
    save_json,
    Timer,
)
from utils.logger import logger


class CorpusBuilder:
    """Assembles balanced multi-domain corpora according to DatasetMixConfig."""

    def __init__(self, config: DatasetMixConfig):
        self.config = config
        self.downloader = DatasetStreamDownloader(config.preprocessing)

    def build_all(self) -> dict[str, Any]:
        """Builds both the 250 MB training corpus and the held-out evaluation corpus."""
        train_path = Path(self.config.corpus.train_output_path)
        eval_path = Path(self.config.corpus.eval_output_path)
        ensure_dir(train_path.parent)
        ensure_dir(eval_path.parent)

        logger.info(
            f"🏗️ Building multi-source corpus: Target {self.config.corpus.total_target_mb} MB "
            f"Train, {self.config.corpus.eval_target_mb} MB Eval"
        )

        source_stats: dict[str, dict[str, Any]] = {}
        train_docs: list[str] = []
        eval_docs: list[str] = []

        # Proportional eval split (e.g. 5 MB total eval divided across sources)
        total_train_mb = self.config.corpus.total_target_mb
        total_eval_mb = self.config.corpus.eval_target_mb

        with Timer("Multi-Source Corpus Acquisition"):
            for source_name, source_cfg in self.config.sources.items():
                if not source_cfg.enabled:
                    logger.info(f"Skipping disabled source: '{source_name}'")
                    continue

                eval_share_mb = (source_cfg.target_mb / total_train_mb) * total_eval_mb
                eval_share_bytes = int(eval_share_mb * 1024 * 1024)
                train_share_bytes = source_cfg.target_bytes

                logger.info(
                    f"📦 Collecting '{source_name}': Train target {format_bytes(train_share_bytes)}, "
                    f"Eval target {format_bytes(eval_share_bytes)}"
                )

                source_train_bytes = 0
                source_eval_bytes = 0
                source_train_count = 0
                source_eval_count = 0

                try:
                    for doc in self.downloader.stream_source_documents(
                        source_cfg, max_bytes=train_share_bytes + eval_share_bytes
                    ):
                        doc_bytes = len(doc.encode("utf-8")) + 2  # including \n\n

                        # First allocate to held-out eval set
                        if source_eval_bytes < eval_share_bytes:
                            eval_docs.append(doc)
                            source_eval_bytes += doc_bytes
                            source_eval_count += 1
                        elif source_train_bytes < train_share_bytes:
                            train_docs.append(doc)
                            source_train_bytes += doc_bytes
                            source_train_count += 1
                        else:
                            break

                    source_stats[source_name] = {
                        "hf_dataset": source_cfg.hf_dataset,
                        "train_docs": source_train_count,
                        "train_bytes": source_train_bytes,
                        "eval_docs": source_eval_count,
                        "eval_bytes": source_eval_bytes,
                    }

                except Exception as e:
                    logger.error(f"Error while acquiring source '{source_name}': {e}")
                    raise CorpusBuildError(f"Failed building source '{source_name}': {e}") from e

        # Deterministic shuffle to blend domains uniformly
        logger.info(f"🔀 Shuffling {len(train_docs):,} train documents with seed={self.config.corpus.seed}...")
        rng = random.Random(self.config.corpus.seed)
        rng.shuffle(train_docs)
        rng.shuffle(eval_docs)

        # Write training corpus
        logger.info(f"✍️ Writing assembled training corpus to {train_path.resolve()}...")
        actual_train_bytes = 0
        with open(train_path, "w", encoding="utf-8") as f:
            for doc in train_docs:
                entry = doc + "\n\n"
                f.write(entry)
                actual_train_bytes += len(entry.encode("utf-8"))

        # Write held-out evaluation corpus
        logger.info(f"✍️ Writing held-out evaluation corpus to {eval_path.resolve()}...")
        actual_eval_bytes = 0
        with open(eval_path, "w", encoding="utf-8") as f:
            for doc in eval_docs:
                entry = doc + "\n\n"
                f.write(entry)
                actual_eval_bytes += len(entry.encode("utf-8"))

        train_sha256 = compute_file_sha256(train_path)
        eval_sha256 = compute_file_sha256(eval_path)

        manifest = {
            "corpus_name": self.config.corpus.name,
            "train_file": str(train_path),
            "train_size_bytes": actual_train_bytes,
            "train_size_readable": format_bytes(actual_train_bytes),
            "train_doc_count": len(train_docs),
            "train_sha256": train_sha256,
            "eval_file": str(eval_path),
            "eval_size_bytes": actual_eval_bytes,
            "eval_size_readable": format_bytes(actual_eval_bytes),
            "eval_doc_count": len(eval_docs),
            "eval_sha256": eval_sha256,
            "sources": source_stats,
        }

        manifest_path = train_path.parent / "corpus_manifest.json"
        save_json(manifest, manifest_path)
        logger.info(f"📜 Saved corpus manifest to {manifest_path.resolve()}")

        return manifest
