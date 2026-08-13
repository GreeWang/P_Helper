"""Stable internal data contracts shared by the pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


MISSING_ZH = "原文未明确说明"
MISSING_EN = "Not explicitly stated in the paper"
ShortText = Annotated[str, Field(min_length=1, max_length=2_000)]
Keyword = Annotated[str, Field(min_length=1, max_length=100)]
EvidenceId = Annotated[str, Field(min_length=1, max_length=200)]


class LLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Page(BaseModel):
    number: int = Field(ge=1)
    text: str
    headings: list[str] = Field(default_factory=list)
    blocks: list["DocumentBlock"] = Field(default_factory=list)


class DocumentBlock(BaseModel):
    id: str
    kind: Literal["heading", "text", "table"]
    page: int = Field(ge=1)
    text: str


class Figure(BaseModel):
    id: str
    page: int = Field(ge=1)
    caption: str = ""
    cache_path: str


class ParsedDocument(BaseModel):
    pages: list[Page]
    figures: list[Figure] = Field(default_factory=list)
    parser_version: str


class Chunk(BaseModel):
    id: str
    pages: list[int]
    text: str


class Fact(LLMModel):
    id: str
    category: Literal[
        "metadata", "problem", "method", "contribution", "experiment",
        "result", "limitation", "keyword", "figure"
    ]
    text: ShortText
    pages: list[Annotated[int, Field(ge=1)]] = Field(min_length=1, max_length=1)
    quote: str = Field(min_length=1, max_length=1000)
    figure_ids: list[EvidenceId] = Field(default_factory=list, max_length=20)


class FactBatch(LLMModel):
    facts: list[Fact] = Field(default_factory=list, max_length=100)


class Claim(LLMModel):
    text: ShortText
    evidence_ids: list[EvidenceId] = Field(default_factory=list, max_length=20)


class ExperimentDesign(LLMModel):
    datasets: Claim
    baselines: Claim
    metrics: Claim
    setup: Claim


class PaperSummary(LLMModel):
    title: Claim
    authors: Claim
    publication_date: Claim
    venue: Claim
    one_sentence: Claim
    research_problem: Claim
    core_method: Claim
    contributions: list[Claim] = Field(min_length=1, max_length=12)
    experiment_design: ExperimentDesign
    main_results: list[Claim] = Field(min_length=1, max_length=12)
    limitations: list[Claim] = Field(min_length=1, max_length=12)
    keywords: list[Claim] = Field(min_length=1, max_length=20)


class SelectedFigure(BaseModel):
    caption: str
    page: int
    relative_path: str


class RetrievalChunk(BaseModel):
    id: str
    fingerprint: str
    source_path: str
    page: int = Field(ge=1)
    headings: list[str] = Field(default_factory=list)
    kind: Literal["heading", "text", "table", "figure"]
    text: str = Field(min_length=1, max_length=2_000)


class QueryExpansion(LLMModel):
    standalone_question: str = Field(min_length=1, max_length=20_000)
    keywords: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(min_length=3, max_length=12)


class QAClaim(LLMModel):
    text: str = Field(min_length=1, max_length=4_000)
    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=20)
    is_inference: bool = False


class QAAnswer(LLMModel):
    sufficient: bool
    claims: list[QAClaim] = Field(default_factory=list, max_length=20)
    suggestions: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(default_factory=list, max_length=12)
