from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


JOB_EXTENSIONS = {".txt", ".md", ".html", ".htm"}
CV_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
SKILL_ALIASES = {
    "postgresql": ("postgresql", "postgres"),
    "kubernetes": ("kubernetes", "k8s"),
    "aws": ("amazon web services", "aws"),
    "gcp": ("google cloud platform", "gcp"),
    "azure": ("microsoft azure", "azure"),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript", "ts"),
    "node.js": ("node.js", "nodejs"),
    "dotnet": (".net", "dotnet"),
    "ci/cd": ("ci/cd", "continuous integration", "continuous delivery"),
}
_NAVIGATION = {
    "skip to content", "sign in", "log in", "menu", "home", "search jobs",
    "share this job", "apply now", "accept cookies", "manage cookies",
}


@dataclass(frozen=True)
class InputDocument:
    path: Path
    raw_bytes: bytes
    content: str
    text: str
    modified_at: datetime

    @property
    def exact_hash(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @property
    def normalized_hash(self) -> str:
        return normalized_hash(self.text)


def discover_job_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise NotADirectoryError(f"Job input path is not a directory: {directory}")
    return sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in JOB_EXTENSIONS),
        key=lambda path: path.as_posix().casefold(),
    )


def read_job_file(path: Path) -> InputDocument:
    if path.suffix.lower() not in JOB_EXTENSIONS:
        raise ValueError(f"Unsupported job file format: {path.suffix}")
    raw_bytes = path.read_bytes()
    raw = raw_bytes.decode("utf-8", errors="replace")
    if path.suffix.lower() in {".html", ".htm"}:
        text = html_to_text(raw)
    else:
        text = raw
    text = clean_job_text(text)
    return InputDocument(path, raw_bytes, raw, text, datetime.fromtimestamp(path.stat().st_mtime))


def read_cv(path: Path, content: bytes | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix not in CV_EXTENSIONS:
        raise ValueError(f"Unsupported CV format: {suffix}")
    data = content if content is not None else path.read_bytes()
    if suffix in {".txt", ".md"}:
        return data.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from io import BytesIO

        pages = PdfReader(BytesIO(data)).pages
        text = "\n".join(page.extract_text() or "" for page in pages)
    else:
        from io import BytesIO

        document = Document(BytesIO(data))
        paragraphs = [paragraph.text for paragraph in document.paragraphs]
        tables = [
            " ".join(cell.text for row in table.rows for cell in row.cells)
            for table in document.tables
        ]
        text = "\n".join([*paragraphs, *tables])
    cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not cleaned:
        raise ValueError(f"No readable text found in CV: {path.name}")
    return cleaned


def html_to_text(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer"]):
        element.decompose()
    return unescape(soup.get_text("\n", strip=True))


def clean_job_text(text: str) -> str:
    lines: list[str] = []
    previous = ""
    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        normalized = re.sub(r"\s+", " ", line).strip()
        if not normalized:
            if lines and lines[-1]:
                lines.append("")
            previous = ""
            continue
        if normalized.casefold() in _NAVIGATION or normalized == previous:
            continue
        lines.append(normalized)
        previous = normalized
    return "\n".join(lines).strip()


def normalized_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"https?://\S+", " ", normalized)
    normalized = re.sub(
        r"\b(?:cookie|privacy|sign in|log in|apply now|skip to content)\b",
        " ",
        normalized,
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonical_skill(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name).casefold().strip()
    for canonical, aliases in SKILL_ALIASES.items():
        if normalized in aliases:
            return canonical
    return normalized


def split_text(text: str, max_chars: int, overlap: int = 500) -> list[str]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    if len(text) <= max_chars:
        return [text]
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end])
                if end == len(paragraph):
                    break
                start = end - overlap
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > max_chars and current:
            chunks.append(current)
            carry = current[-overlap:] if overlap else ""
            current = f"{carry}\n\n{paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def extract_possible_urls(text: str) -> list[str]:
    found = re.findall(r"https?://[^\s<>\"')]+", text)
    unique: list[str] = []
    seen: set[str] = set()
    for value in found:
        parsed = urlparse(value)
        if parsed.scheme in {"http", "https"} and value not in seen:
            unique.append(value.rstrip(".,;"))
            seen.add(value)
    return unique
