"""从 SKILL.md 加载 Agent 系统提示，并拼接会话动态上下文。"""

from __future__ import annotations

from pathlib import Path

from core.state import SessionState

_DEFAULT_SKILL = Path(__file__).resolve().parent / "tools" / "SKILL.md"


def load_skill_markdown_body(skill_path: Path | None = None) -> str:
    """读取 SKILL.md，去掉 YAML frontmatter（首段 --- ... ---），返回正文。"""
    path = skill_path or _DEFAULT_SKILL
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("---"):
        return text.strip()
    parts = text.split("---", 2)
    if len(parts) >= 3:
        return parts[2].lstrip("\n\r").strip()
    return text.strip()


def format_session_digest(session: SessionState) -> str:
    lines = [
        f"- workflow_phase: {session.workflow_phase}",
        f"- style_preference: {session.style_preference or '（未设置）'}",
        f"- room_image_url: {session.room_image_url or '（未设置）'}",
        f"- room_style_analysis: {session.room_style_analysis or '（未分析）'}",
        f"- 决策池家具件数: {len(session.user_list)}",
        f"- 决策池总价 user_list_total: {session.user_list_total:.2f} 元",
    ]
    if session.user_list:
        lines.append("- User-List 明细（id / name / category / price / style_tags / material / dimensions）:")
        for it in session.user_list:
            dims = it.dimensions
            dim_text = f"{dims.w:.0f}×{dims.d:.0f}×{dims.h:.0f} cm"
            style_text = "、".join(it.style_tags) if it.style_tags else "—"
            lines.append(
                "  - "
                f"{it.id} | name={it.name or '（无名称）'} | category={it.category.value} "
                f"| price={it.price:.2f} | style_tags={style_text} "
                f"| material={it.material or '—'} | dimensions={dim_text}"
            )
            if it.description:
                lines.append(f"    description={it.description}")
    else:
        lines.append("- User-List: （空）")
    if session.workflow_phase == "finished":
        lines.append(
            "- 用户已确认清单完成：除非用户明确要求，请勿调用 show_list_add；"
            "不要编造或擅自修改 User-List。"
        )
    return "\n".join(lines)


def build_system_prompt(
    session: SessionState,
    *,
    skill_path: Path | None = None,
) -> str:
    base = load_skill_markdown_body(skill_path)
    digest = format_session_digest(session)
    return (
        f"{base}\n\n---\n## 当前会话状态（动态，每轮更新）\n\n{digest}"
    )
