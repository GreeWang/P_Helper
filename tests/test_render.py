from frame.models import (Claim, ExperimentDesign, Fact, PaperSummary,
                          SelectedFigure)
from frame.render import render_paper


def test_chinese_rendering_has_fixed_sections_and_physical_pages():
    fact = Fact(id="f1", category="result", text="Accuracy 91%", pages=[2], quote="Accuracy 91%")
    claim = Claim(text="准确率达到 91%。", evidence_ids=["f1"])
    missing = Claim(text="原文未明确说明", evidence_ids=[])
    summary = PaperSummary(
        title=Claim(text="Original Title", evidence_ids=["f1"]),
        authors=Claim(text="原文未明确说明", evidence_ids=[]),
        publication_date=Claim(text="原文未明确说明", evidence_ids=[]),
        venue=Claim(text="原文未明确说明", evidence_ids=[]),
        one_sentence=claim, research_problem=claim, core_method=claim,
        contributions=[claim], experiment_design=ExperimentDesign(
            datasets=missing, baselines=missing, metrics=claim, setup=missing),
        main_results=[claim], limitations=[missing], keywords=[claim],
    )
    text = render_paper(summary, [fact], [SelectedFigure(
        caption="Architecture", page=3, relative_path="images/x/figure-01.png")],
        "source.pdf", "model-x", "zh")
    headings = ["基本信息", "一句话总结", "研究问题", "核心方法", "主要贡献",
                "实验设计", "主要结果", "局限性", "代表图片", "关键词"]
    assert [text.index(f"## {item}") for item in headings] == sorted(
        text.index(f"## {item}") for item in headings)
    assert "PDF 第 2 页" in text
    assert "../images/x/figure-01.png" in text


def test_english_rendering_switches_the_template():
    fact = Fact(id="f1", category="method", text="Method", pages=[1], quote="Method")
    claim = Claim(text="Method", evidence_ids=["f1"])
    missing = Claim(text="Not explicitly stated in the paper", evidence_ids=[])
    summary = PaperSummary(
        title=Claim(text="T", evidence_ids=["f1"]), authors=missing,
        publication_date=missing, venue=missing, one_sentence=claim,
        research_problem=claim, core_method=claim, contributions=[claim],
        experiment_design=ExperimentDesign(datasets=missing, baselines=missing, metrics=missing, setup=missing),
        main_results=[missing], limitations=[missing], keywords=[claim])
    text = render_paper(summary, [fact], [], "s.pdf", "m", "en")
    assert "## Basic Information" in text
    assert "PDF pages 1" in text
