"""Deterministic Markdown rendering from validated models."""

from .models import Claim, Fact, PaperSummary, SelectedFigure


def render_paper(summary: PaperSummary, facts: list[Fact], figures: list[SelectedFigure],
                 source: str, model: str, language: str) -> str:
    by_id = {fact.id: fact for fact in facts}
    if language == "en":
        return _render_en(summary, by_id, figures, source, model)
    return _render_zh(summary, by_id, figures, source, model)


def _pages(claim: Claim, facts: dict[str, Fact], language: str) -> str:
    pages = sorted({page for item in claim.evidence_ids for page in facts[item].pages})
    if not pages:
        return ""
    label = ", ".join(str(page) for page in pages)
    return f" (PDF pages {label})" if language == "en" else f"（PDF 第 {label} 页）"


def _claim(claim: Claim, facts: dict[str, Fact], language: str) -> str:
    return claim.text + _pages(claim, facts, language)


def _items(claims: list[Claim], facts: dict[str, Fact], language: str) -> str:
    return "\n".join(f"{index}. {_claim(claim, facts, language)}"
                     for index, claim in enumerate(claims, 1))


def _render_zh(s, facts, figures, source, model):
    figure_text = "\n\n".join(
        f"![{figure.caption or '代表图片'}](../{figure.relative_path})\n\n"
        f"{figure.caption or '原文未提供图片说明'}（PDF 第 {figure.page} 页）"
        for figure in figures
    ) or "原文未明确说明"
    return f"""# {s.title.text}

## 基本信息

- 原始文件：`{source}`
- 作者：{s.authors.text}
- 发表时间：{s.publication_date.text}
- 会议 / 期刊：{s.venue.text}
- 处理模型：`{model}`

## 一句话总结

{_claim(s.one_sentence, facts, 'zh')}

## 研究问题

{_claim(s.research_problem, facts, 'zh')}

## 核心方法

{_claim(s.core_method, facts, 'zh')}

## 主要贡献

{_items(s.contributions, facts, 'zh')}

## 实验设计

- 数据集：{_claim(s.experiment_design.datasets, facts, 'zh')}
- 对比方法：{_claim(s.experiment_design.baselines, facts, 'zh')}
- 评价指标：{_claim(s.experiment_design.metrics, facts, 'zh')}
- 实验设置：{_claim(s.experiment_design.setup, facts, 'zh')}

## 主要结果

{_items(s.main_results, facts, 'zh')}

## 局限性

{_items(s.limitations, facts, 'zh')}

## 代表图片

{figure_text}

## 关键词

{'、'.join(f'`{item.text}`' for item in s.keywords)}
"""


def _render_en(s, facts, figures, source, model):
    figure_text = "\n\n".join(
        f"![{figure.caption or 'Representative figure'}](../{figure.relative_path})\n\n"
        f"{figure.caption or 'No caption provided'} (PDF page {figure.page})"
        for figure in figures
    ) or "Not explicitly stated in the paper"
    return f"""# {s.title.text}

## Basic Information

- Source file: `{source}`
- Authors: {s.authors.text}
- Publication date: {s.publication_date.text}
- Venue: {s.venue.text}
- Processing model: `{model}`

## One-sentence Summary

{_claim(s.one_sentence, facts, 'en')}

## Research Problem

{_claim(s.research_problem, facts, 'en')}

## Core Method

{_claim(s.core_method, facts, 'en')}

## Main Contributions

{_items(s.contributions, facts, 'en')}

## Experimental Design

- Datasets: {_claim(s.experiment_design.datasets, facts, 'en')}
- Baselines: {_claim(s.experiment_design.baselines, facts, 'en')}
- Metrics: {_claim(s.experiment_design.metrics, facts, 'en')}
- Setup: {_claim(s.experiment_design.setup, facts, 'en')}

## Main Results

{_items(s.main_results, facts, 'en')}

## Limitations

{_items(s.limitations, facts, 'en')}

## Representative Figures

{figure_text}

## Keywords

{', '.join(f'`{item.text}`' for item in s.keywords)}
"""
