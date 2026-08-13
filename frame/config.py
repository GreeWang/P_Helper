"""Validated runtime configuration for the batch pipeline."""

from dataclasses import dataclass


DEFAULT_OUTPUT_DIR = "summaries"
DEFAULT_CACHE_MAX_PAPERS = 50
DEFAULT_CACHE_MAX_SIZE = 5 * 1024 ** 3


@dataclass(frozen=True)
class Config:
    api_key: str
    api_url: str
    model: str
    output_dir: str = DEFAULT_OUTPUT_DIR
    language: str = "zh"
    workers: int = 1
    force: bool = False
    cache_max_papers: int = DEFAULT_CACHE_MAX_PAPERS
    cache_max_size: int = DEFAULT_CACHE_MAX_SIZE
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_backoff: float = 5.0
    json_repair_attempts: int = 2

    def __post_init__(self):
        if self.language not in {"zh", "en"}:
            raise ValueError("language must be 'zh' or 'en'")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.cache_max_papers < 0 or self.cache_max_size < 0:
            raise ValueError("cache limits cannot be negative")

