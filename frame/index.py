"""Rebuild the human-readable batch index from durable state."""

from .manifest import Manifest


def render_index(manifest: Manifest, language: str) -> str:
    entries = sorted(manifest.papers.values(), key=lambda item: item.source_path)
    if language == "en":
        lines = [
            "# P-Helper Paper Summaries", "",
            "| Source | Summary | Summary status | QA index | Detail |",
            "|---|---|---|---|---|",
        ]
        labels = {"success": "Success", "failed": "Failed", "processing": "Processing"}
    else:
        lines = [
            "# P-Helper 论文摘要", "",
            "| 来源 | 摘要 | 摘要状态 | 问答索引 | 说明 |",
            "|---|---|---|---|---|",
        ]
        labels = {"success": "成功", "failed": "失败", "processing": "处理中"}
    for entry in entries:
        source = _cell(entry.source_path)
        link = f"[打开]({entry.output_path})" if entry.output_path and entry.status == "success" else "-"
        details = []
        if entry.error:
            details.append(("summary: " if language == "en" else "摘要：") + entry.error)
        if entry.index_error:
            details.append(("index: " if language == "en" else "索引：") + entry.index_error)
        detail = _cell("; ".join(details) or entry.stage)
        index_labels = ({"not_run": "Not indexed", "processing": "Indexing",
                         "success": "Ready", "failed": "Failed"} if language == "en" else
                        {"not_run": "未建立", "processing": "建立中",
                         "success": "可用", "failed": "失败"})
        lines.append(
            f"| `{source}` | {link} | {labels[entry.status]} | "
            f"{index_labels[entry.index_status]} | {detail} |"
        )
    return "\n".join(lines) + "\n"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
