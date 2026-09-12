"""Pydantic configuration models and schema validation for the BPE Tokenizer project."""

from pathlib import Path
from typing import Any
import regex as re
from pydantic import BaseModel, Field, field_validator, model_validator

from utils.custom_exception import ConfigValidationError
from utils.helpers import load_yaml


# ==========================================
# 1. Base & System Configuration
# ==========================================
class SystemPathsConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    experiments_dir: Path = Path("experiments/runs")
    visuals_dir: Path = Path("experiments/visuals")
    logs_dir: Path = Path("logs")


class SystemSettingsConfig(BaseModel):
    seed: int = 42
    num_workers: int = 4
    log_level: str = "INFO"


class BaseConfig(BaseModel):
    system: SystemSettingsConfig = Field(default_factory=SystemSettingsConfig)
    paths: SystemPathsConfig = Field(default_factory=SystemPathsConfig)


# ==========================================
# 2. Dataset Mix Configuration
# ==========================================
class SourceConfig(BaseModel):
    enabled: bool = True
    hf_dataset: str
    subset: str | None = None
    split: str = "train"
    target_mb: float = Field(gt=0, description="Target size in megabytes")
    text_column: str = "text"
    description: str = ""

    @property
    def target_bytes(self) -> int:
        return int(self.target_mb * 1024 * 1024)


class CorpusConfig(BaseModel):
    name: str = "gpt_general_purpose_250mb"
    total_target_mb: float = Field(gt=0, default=250.0)
    train_output_path: Path = Path("data/processed/train_corpus_250mb.txt")
    eval_output_path: Path = Path("data/processed/eval_corpus_heldout.txt")
    eval_target_mb: float = Field(gt=0, default=5.0)
    seed: int = 42
    shuffle_buffer_size: int = 50000

    @property
    def train_target_bytes(self) -> int:
        return int(self.total_target_mb * 1024 * 1024)

    @property
    def eval_target_bytes(self) -> int:
        return int(self.eval_target_mb * 1024 * 1024)


class PreprocessingConfig(BaseModel):
    preserve_casing: bool = True
    preserve_whitespace: bool = True
    preserve_numbers: bool = True
    preserve_urls: bool = True
    remove_null_bytes: bool = True
    normalize_newlines: bool = True
    min_document_length_chars: int = 50


class DatasetMixConfig(BaseModel):
    corpus: CorpusConfig
    sources: dict[str, SourceConfig]
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @model_validator(mode="after")
    def validate_proportions(self) -> "DatasetMixConfig":
        enabled_total = sum(s.target_mb for s in self.sources.values() if s.enabled)
        # Allow slight rounding variance (0.5 MB)
        if abs(enabled_total - self.corpus.total_target_mb) > 0.5:
            raise ValueError(
                f"Enabled source target MB sum ({enabled_total:.2f} MB) does not match "
                f"corpus total_target_mb ({self.corpus.total_target_mb:.2f} MB)"
            )
        return self


# ==========================================
# 3. Tokenizer Configuration
# ==========================================
class SpecialTokensConfig(BaseModel):
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    unk_token: str = "<|unk|>"
    bos_token: str = "<|bos|>"
    additional_special_tokens: list[str] = Field(default_factory=list)

    @property
    def all_special_tokens(self) -> list[str]:
        tokens = [self.eos_token, self.pad_token, self.unk_token, self.bos_token]
        for t in self.additional_special_tokens:
            if t not in tokens:
                tokens.append(t)
        return tokens


class TokenizerConfig(BaseModel):
    name: str = "general_purpose_bpe_32k"
    model_type: str = "byte_level_bpe"
    vocab_size: int = Field(ge=256, default=32000)
    min_frequency: int = Field(ge=1, default=2)
    byte_level: bool = True
    regex_pattern: str
    special_tokens: SpecialTokensConfig = Field(default_factory=SpecialTokensConfig)

    @field_validator("regex_pattern")
    @classmethod
    def validate_regex(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid pre-tokenization regex pattern: {e}") from e
        return pattern


class TokenizerConfigWrapper(BaseModel):
    tokenizer: TokenizerConfig


# ==========================================
# 4. Downstream LM Evaluation Configuration
# ==========================================
class LMModelConfig(BaseModel):
    block_size: int = Field(default=512, ge=64)
    n_layer: int = Field(default=6, ge=1)
    n_head: int = Field(default=6, ge=1)
    n_embd: int = Field(default=384, ge=64)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)
    bias: bool = False


class LMTrainingConfig(BaseModel):
    batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=6.0e-4, gt=0)
    min_lr: float = Field(default=6.0e-5, gt=0)
    weight_decay: float = Field(default=0.1, ge=0.0)
    beta1: float = 0.9
    beta2: float = 0.95
    max_steps: int = Field(default=2000, ge=100)
    eval_interval: int = Field(default=200, ge=10)
    eval_steps: int = Field(default=40, ge=5)
    warmup_steps: int = Field(default=100, ge=0)
    device: str = "auto"
    seed: int = 42


class LMDatasetConfig(BaseModel):
    train_split_ratio: float = Field(default=0.90, gt=0.5, lt=1.0)
    max_tokens_eval: int = Field(default=2000000, ge=10000)


class LMEvalConfig(BaseModel):
    model: LMModelConfig = Field(default_factory=LMModelConfig)
    training: LMTrainingConfig = Field(default_factory=LMTrainingConfig)
    dataset: LMDatasetConfig = Field(default_factory=LMDatasetConfig)


class LMEvalConfigWrapper(BaseModel):
    lm_evaluation: LMEvalConfig


# ==========================================
# Loader Functions with Strict Validation
# ==========================================
def load_base_config(file_path: str | Path = "configs/base_config.yaml") -> BaseConfig:
    raw = load_yaml(file_path)
    try:
        return BaseConfig.model_validate(raw)
    except Exception as e:
        raise ConfigValidationError(f"BaseConfig validation failed: {e}") from e


def load_dataset_mix_config(file_path: str | Path = "configs/dataset_mix.yaml") -> DatasetMixConfig:
    raw = load_yaml(file_path)
    try:
        return DatasetMixConfig.model_validate(raw)
    except Exception as e:
        raise ConfigValidationError(f"DatasetMixConfig validation failed: {e}") from e


def load_tokenizer_config(file_path: str | Path = "configs/tokenizer_32k.yaml") -> TokenizerConfig:
    raw = load_yaml(file_path)
    try:
        return TokenizerConfigWrapper.model_validate(raw).tokenizer
    except Exception as e:
        raise ConfigValidationError(f"TokenizerConfig validation failed: {e}") from e


def load_lm_eval_config(file_path: str | Path = "configs/lm_eval_config.yaml") -> LMEvalConfig:
    raw = load_yaml(file_path)
    try:
        return LMEvalConfigWrapper.model_validate(raw).lm_evaluation
    except Exception as e:
        raise ConfigValidationError(f"LMEvalConfig validation failed: {e}") from e
