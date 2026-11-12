from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests import RequestException

from career_agent.config import ResearchSettings
from career_agent.ingestion import clean_job_text

USER_AGENT = "CareerIntelligenceAgent/0.1 (public research; no login or automation)"


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str


@dataclass(frozen=True)
class ResearchPage:
    url: str
    title: str
    text: str
    retrieved_at: datetime
    category: str = "general"


class ResearchTool:
    def search(self, query: str, limit: int, timeout: float) -> list[SearchResult]:
        raise NotImplementedError

    def fetch(self, url: str, timeout: float) -> ResearchPage | None:
        raise NotImplementedError


class DuckDuckGoResearchTool(ResearchTool):
    """Free HTML search and public-page retrieval without login or CAPTCHA handling."""

    def search(self, query: str, limit: int = 5, timeout: float = 10) -> list[SearchResult]:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[SearchResult] = []
        for anchor in soup.select("a.result__a"):
            href = anchor.get("href", "")
            parsed = urlparse(href)
            if parsed.hostname and "duckduckgo.com" in parsed.hostname:
                target = parse_qs(parsed.query).get("uddg", [None])[0]
                href = target or href
            if not _is_public_web_url(href):
                continue
            results.append(SearchResult(href, anchor.get_text(" ", strip=True)))
            if len(results) >= limit:
                break
        return results

    def fetch(self, url: str, timeout: float = 10) -> ResearchPage | None:
        _require_public_web_url(url)
        if not _robots_allow(url, timeout):
            return None
        response = _safe_get(url, timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").casefold()
        if content_type and not any(
            accepted in content_type for accepted in ("text/html", "text/plain", "application/xhtml")
        ):
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer"]):
            element.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = clean_job_text(soup.get_text("\n", strip=True))
        if not text:
            return None
        return ResearchPage(
            url=response.url,
            title=title[:300],
            text=text[:30000],
            retrieved_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )


def build_research_queries(company: str) -> list[tuple[str, str]]:
    safe_name = " ".join(company.split())[:120]
    return [
        (f"{safe_name} official products business locations", "company"),
        (f"{safe_name} engineering blog technology", "engineering"),
        (f"{safe_name} GitHub engineering projects", "public_projects"),
        (f"{safe_name} engineering team practices recent news", "engineering"),
        (f"{safe_name} company size stability", "company"),
    ]


def _is_public_web_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password or parsed.port not in {None, 80, 443}:
            return False
        if parsed.hostname.casefold() in {"localhost", "localhost.localdomain"}:
            return False
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM)
        return bool(addresses) and all(ipaddress.ip_address(row[4][0]).is_global for row in addresses)
    except (ValueError, OSError):
        return False


def _require_public_web_url(url: str) -> None:
    if not _is_public_web_url(url):
        raise ValueError(f"Research URL is not a public HTTP(S) address: {url}")


def _safe_get(url: str, timeout: float, redirects_left: int = 3) -> requests.Response:
    _require_public_web_url(url)
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        allow_redirects=False,
    )
    if response.is_redirect or response.is_permanent_redirect:
        if redirects_left <= 0:
            raise RequestException(f"Too many redirects while fetching {url}")
        location = response.headers.get("location")
        if not location:
            raise RequestException(f"Redirect from {url} did not provide a location")
        response.close()
        return _safe_get(urljoin(url, location), timeout, redirects_left - 1)
    return response


def _robots_allow(url: str, timeout: float) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = _safe_get(robots_url, timeout)
        if response.status_code == 404:
            return True
        response.raise_for_status()
        parser.parse(response.text.splitlines())
        return parser.can_fetch(USER_AGENT, url)
    except RequestException:
        return False
