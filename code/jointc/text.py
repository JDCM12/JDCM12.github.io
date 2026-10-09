from __future__ import annotations

import re
from collections import Counter
from typing import Any


TEXT_SPECIAL_TOKENS = [
    "[prompt_start]",
    "[prompt_end]",
    "[lyrics_start]",
    "[lyrics_end]",
]

_NEWLINE = re.compile(r"\r\n|\r|\n")
_SPACE = re.compile(r"[ \t\f\v]+")
_SEMICOLON_SPACE = re.compile(r"\s*;\s*")
_MULTI_SEMICOLON = re.compile(r";{2,}")
_BRACKET = re.compile(r"\[([^\[\]\n]{1,160})\]")
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")

_SECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"pre\s*[- _]?\s*chorus"), "pre_chorus"),
    (re.compile(r"post\s*[- _]?\s*chorus"), "post_chorus"),
    (re.compile(r"chorus"), "chorus"),
    (re.compile(r"verse"), "verse"),
    (re.compile(r"intro"), "intro"),
    (re.compile(r"outro"), "outro"),
    (re.compile(r"bridge"), "bridge"),
    (re.compile(r"hook"), "hook"),
    (re.compile(r"refrain"), "refrain"),
    (re.compile(r"interlude"), "interlude"),
    (re.compile(r"instrumental"), "instrumental"),
    (re.compile(r"solo"), "solo"),
    (re.compile(r"breakdown"), "breakdown"),
    (re.compile(r"build\s*[- _]?\s*up"), "build_up"),
    (re.compile(r"drop"), "drop"),
]
_NUMBERED_SECTIONS = {"verse", "chorus", "pre_chorus", "post_chorus"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = _NEWLINE.sub(";", text)
    text = _SEMICOLON_SPACE.sub(";", text)
    text = _MULTI_SEMICOLON.sub(";", text)
    text = _SPACE.sub(" ", text)
    return text.strip(" ;")


def format_structure(structure: Any) -> str:
    if structure is None:
        return ""
    if isinstance(structure, str):
        return normalize_text(structure)
    return ";".join(
        normalized
        for item in structure
        if (normalized := normalize_text(item))
    )


def canonical_section(raw_tag: str) -> str | None:
    tag = raw_tag.strip().lower()
    if not tag:
        return None
    tag = re.sub(r"\([^)]*\)", " ", tag)
    tag = tag.replace("_", " ")
    tag = re.sub(r"\s+", " ", tag).strip()
    for pattern, canonical in _SECTION_PATTERNS:
        if pattern.search(tag):
            return canonical
    return None


def _cleanup_structured_lyrics(lyrics: str) -> str:
    lyrics = _TRAILING_SPACE.sub("\n", lyrics)
    lyrics = _MULTI_BLANK.sub("\n\n", lyrics)
    return lyrics.strip()


def number_section_tags(lyrics: Any) -> tuple[str, list[str]]:
    raw_lyrics = "" if lyrics is None else str(lyrics)
    structure: list[str] = []
    counters: Counter[str] = Counter()

    def replace_tag(match: re.Match[str]) -> str:
        section = canonical_section(match.group(1))
        if section is None:
            return "\n"
        if section in _NUMBERED_SECTIONS:
            counters[section] += 1
            label = f"{section}_{counters[section]}"
        else:
            label = section
        structure.append(label)
        return f"\n[{label}]\n"

    numbered = _BRACKET.sub(replace_tag, raw_lyrics)
    return _cleanup_structured_lyrics(numbered), structure


def build_text_condition(
    prompt: Any,
    lyrics: Any,
    structure: Any = None,
) -> str:
    """Build the v2 text condition used by the JointC Qwen backbone."""

    prompt_text = normalize_text(prompt)
    numbered_lyrics, inline_structure = number_section_tags(lyrics)
    structured_lyrics = normalize_text(numbered_lyrics)
    fallback_lyrics = normalize_text(lyrics)

    if inline_structure or structured_lyrics == fallback_lyrics:
        lyric_condition = structured_lyrics
    else:
        fallback_structure = format_structure(structure)
        lyric_condition = ";".join(
            part for part in (fallback_structure, fallback_lyrics) if part
        )

    parts: list[str] = []
    if prompt_text:
        parts.append(f"[prompt_start] {prompt_text} [prompt_end]")
    if lyric_condition:
        parts.append(f"[lyrics_start] {lyric_condition} [lyrics_end]")
    return ";".join(parts)
