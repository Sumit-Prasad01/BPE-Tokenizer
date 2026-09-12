"""High-performance Byte-Level BPE training engine using an inverted index and priority heap."""

import heapq
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from tqdm import tqdm

from src.tokenizer.byte_encoder import bytes_to_unicode
from src.tokenizer.pre_tokenizer import RegexPreTokenizer
from utils.custom_exception import BpeTrainingError
from utils.helpers import Timer, format_bytes, format_number
from utils.logger import logger


class BPETrainer:
    """Trains a Byte-Level Byte Pair Encoding (BPE) vocabulary and merge rules from a text corpus."""

    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
        regex_pattern: str | None = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "<|endoftext|>",
            "<|pad|>",
            "<|unk|>",
            "<|bos|>",
            "<|eos|>",
        ]
        self.pre_tokenizer = RegexPreTokenizer(regex_pattern)
        self.byte_encoder = bytes_to_unicode()

    def train_from_file(
        self,
        corpus_path: str | Path,
        max_lines: int | None = None,
    ) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, Any]]:
        """Trains BPE vocabulary and merge table directly from a text file.
        
        Returns:
            - vocab: dict mapping token string -> token ID
            - merges: list of ordered (token_a, token_b) merge rules
            - stats: training statistics dictionary
        """
        logger.info(f"🚀 Commencing BPE training targeting {self.vocab_size:,} vocabulary...")
        
        # Step 1: Pre-tokenize and aggregate word frequencies
        word_counts = self.pre_tokenizer.count_word_frequencies(corpus_path, max_lines=max_lines)
        return self.train_from_word_counts(word_counts)

    def train_from_word_counts(
        self,
        word_counts: Counter[tuple[str, ...]],
    ) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, Any]]:
        """Trains BPE merges given pre-tokenized word frequency counts."""
        start_time = time.perf_counter()

        if not word_counts:
            raise BpeTrainingError("Cannot train BPE on an empty word frequency counter.")

        # Initialize base vocabulary: special tokens first, then all 256 byte tokens
        vocab: dict[str, int] = {}
        token_id = 0

        for sp_token in self.special_tokens:
            if sp_token not in vocab:
                vocab[sp_token] = token_id
                token_id += 1

        # Add all 256 raw byte representations
        for byte_val in range(256):
            char_rep = self.byte_encoder[byte_val]
            if char_rep not in vocab:
                vocab[char_rep] = token_id
                token_id += 1

        base_vocab_size = len(vocab)
        num_merges_target = self.vocab_size - base_vocab_size

        logger.info(
            f"Initialized base vocabulary with {base_vocab_size} tokens "
            f"({len(self.special_tokens)} special + 256 bytes). "
            f"Target merges to learn: {num_merges_target:,}"
        )

        if num_merges_target <= 0:
            logger.warning(
                f"Target vocab_size ({self.vocab_size}) <= base vocab size ({base_vocab_size}). "
                "No merges will be trained."
            )
            return vocab, [], {"time_seconds": 0.0, "merges_count": 0}

        # Structure words for fast inverted index updates
        # words: list of list of symbols
        # word_freqs: list of integer frequencies
        words: list[list[str]] = []
        word_freqs: list[int] = []

        for word_tuple, freq in word_counts.items():
            if len(word_tuple) > 1:
                words.append(list(word_tuple))
                word_freqs.append(freq)

        # Build initial pair frequencies and inverted index
        pair_freqs: dict[tuple[str, str], int] = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set[int]] = defaultdict(set)

        logger.info("Building initial inverted index and pair frequency table...")
        for word_id, word in enumerate(words):
            freq = word_freqs[word_id]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word_id)

        # Build max-heap for O(1) retrieval of highest frequency pair
        # Heap stores (-frequency, pair)
        heap: list[tuple[int, tuple[str, str]]] = [
            (-freq, pair) for pair, freq in pair_freqs.items()
        ]
        heapq.heapify(heap)

        merges: list[tuple[str, str]] = []
        merge_frequencies: list[int] = []

        logger.info(f"Learning {num_merges_target:,} BPE merge rules...")
        pbar = tqdm(total=num_merges_target, desc="BPE Merging", unit="merge")

        while len(merges) < num_merges_target and heap:
            neg_freq, best_pair = heapq.heappop(heap)
            current_freq = pair_freqs.get(best_pair, 0)

            # Lazy heap invalidation check: pair frequency might have changed
            if -neg_freq != current_freq:
                if current_freq > 0:
                    heapq.heappush(heap, (-current_freq, best_pair))
                continue

            if current_freq < self.min_frequency:
                logger.info(
                    f"🛑 Frequency {current_freq} fell below min_frequency={self.min_frequency}. "
                    f"Stopping merge process early at {len(merges):,} merges."
                )
                break

            # Register new merge
            a, b = best_pair
            new_token = a + b
            vocab[new_token] = token_id
            token_id += 1
            merges.append(best_pair)
            merge_frequencies.append(current_freq)
            pbar.update(1)

            # Retrieve words containing the winning pair
            affected_word_ids = list(pair_to_words.pop(best_pair, set()))
            pair_freqs.pop(best_pair, None)

            # Update affected words and track changed adjacent pairs
            changed_pairs: set[tuple[str, str]] = set()

            for word_id in affected_word_ids:
                word = words[word_id]
                w_freq = word_freqs[word_id]
                new_word: list[str] = []
                i = 0
                n = len(word)

                while i < n:
                    if i < n - 1 and word[i] == a and word[i + 1] == b:
                        # Decrement old pair with preceding symbol
                        if new_word:
                            old_prev_pair = (new_word[-1], a)
                            pair_freqs[old_prev_pair] -= w_freq
                            changed_pairs.add(old_prev_pair)
                            if pair_freqs[old_prev_pair] <= 0:
                                pair_to_words[old_prev_pair].discard(word_id)

                        # Decrement old pair with succeeding symbol
                        if i + 2 < n:
                            old_next_pair = (b, word[i + 2])
                            pair_freqs[old_next_pair] -= w_freq
                            changed_pairs.add(old_next_pair)
                            if pair_freqs[old_next_pair] <= 0:
                                pair_to_words[old_next_pair].discard(word_id)

                        # Append merged token
                        new_word.append(new_token)

                        # Increment new pair with preceding symbol
                        if len(new_word) > 1:
                            new_prev_pair = (new_word[-2], new_token)
                            pair_freqs[new_prev_pair] += w_freq
                            pair_to_words[new_prev_pair].add(word_id)
                            changed_pairs.add(new_prev_pair)

                        i += 2
                    else:
                        # Before appending word[i], if previous was new_token, increment new pair
                        if new_word and new_word[-1] == new_token and i > 0 and word[i - 1] == b:
                            new_next_pair = (new_token, word[i])
                            pair_freqs[new_next_pair] += w_freq
                            pair_to_words[new_next_pair].add(word_id)
                            changed_pairs.add(new_next_pair)

                        new_word.append(word[i])
                        i += 1

                words[word_id] = new_word

            # Push updated frequencies to the heap
            for pair in changed_pairs:
                freq = pair_freqs.get(pair, 0)
                if freq > 0:
                    heapq.heappush(heap, (-freq, pair))

        pbar.close()
        elapsed_time = time.perf_counter() - start_time

        stats = {
            "training_time_seconds": round(elapsed_time, 2),
            "merges_trained": len(merges),
            "final_vocab_size": len(vocab),
            "unique_words_processed": len(word_counts),
            "merge_frequencies": merge_frequencies,
        }

        logger.info(
            f"🎉 Training finished in {elapsed_time:.2f}s! "
            f"Learned {len(merges):,} merges, final vocab size: {len(vocab):,} tokens."
        )

        return vocab, merges, stats
