"""SerpApi Google Search and Google Scholar citation verification."""
import requests
import openai
import tempfile
import concurrent.futures
import json
import os
from typing import Dict, List, Optional
from pdf_processor import get_page_text
import threading
import time
import serpapi

try:
    from google import genai
except Exception:
    genai = None

try:
    from config import GEMINI_API_KEY as CONFIG_GEMINI_API_KEY
    from config import SERPAPI_KEY as CONFIG_SERPAPI_KEY
except Exception:
    CONFIG_GEMINI_API_KEY = ""
    CONFIG_SERPAPI_KEY = ""

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY") or CONFIG_SERPAPI_KEY or ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or CONFIG_GEMINI_API_KEY
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


def _query_serpapi_account() -> Optional[Dict]:
    """Hit SerpAPI /account.json. Returns dict or None on error. Does not consume search quota."""
    try:
        resp = requests.get(
            "https://serpapi.com/account.json",
            params={"api_key": SERPAPI_KEY},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def initialize_provider_routing():
    """Call once at startup: log SerpAPI quota for visibility."""
    if not SERPAPI_KEY:
        return
    info = _query_serpapi_account()
    if not info:
        print("[provider] SerpAPI /account.json unreachable — proceeding anyway")
        return
    remaining = info.get("total_searches_left", 0)
    plan = info.get("plan_name", "?")
    used = info.get("this_month_usage", "?")
    monthly = info.get("searches_per_month", "?")
    print(f"[provider] SerpAPI plan={plan!r} used={used}/{monthly} remaining={remaining}")


# Global flag to skip SerpAPI if quota is exhausted
SKIP_SERPAPI = False

# Global lock for thread error handling
thread_error_lock = threading.Lock()
thread_error_pause = False


class FatalAPIError(Exception):
    """Unrecoverable API error — caller should abort the whole run."""
    pass


class GeminiFatalError(FatalAPIError):
    pass


class SerpAPIFatalError(FatalAPIError):
    pass


_GEMINI_FATAL_PATTERNS = (
    "api_key_invalid", "api key not valid", "permission_denied",
    "unauthenticated", "unauthorized", " 401", " 403",
)
_SERPAPI_FATAL_PATTERNS = (
    "invalid api key", "your account has run out", "no plan",
    "unauthorized", "401", "403",
)

EXTRACT_PROMPT = """Extract structured metadata from this academic citation.
Return ONLY a JSON object with these fields:
{{"authors": "author names as they appear", "title": "paper/work title", "venue": "journal, conference, publisher, or N/A", "year": "publication year or N/A"}}

Use the provided structured fields when they are already reliable, but fix obvious OCR or formatting issues.

Citation:
{raw}"""

COMPARE_PROMPT = """Decide if the cited paper exists, given the search results below.

CITATION:
Title:   {title}
Authors: {authors}
Venue:   {venue}
Year:    {year}

SEARCH RESULTS:
{documents}

DECIDE only from the SEARCH RESULTS.

==== TITLE ====
First decide what kind of source this is:

(A) ACADEMIC PUBLICATION (peer-reviewed paper, preprint, conference, journal article,
    technical report, thesis, book) — STRICT title matching:
    The title must match the citation EXACTLY at the word level. The ONLY allowed
    differences are case, punctuation, whitespace, and hyphenation between words.
    Any word added, removed, substituted, or reordered → match=false.
    A renamed published version with different title words is a TITLE MISMATCH
    (but if the citation title exactly matches the preprint/earlier version that
    the result also surfaces, that counts as a strict title match).

(B) NON-ACADEMIC ONLINE RESOURCE (blog post, website, dataset card, model card on
    Hugging Face, documentation page, GitHub README, news article, government doc,
    standards document, online dictionary, etc.) — RELAXED title matching:
    The cited "title" is often a section heading, project/feature name, page label,
    or descriptive phrase rather than the HTML <title>. Accept the title as matching
    when the cited title:
      - is a section heading or H1/H2/H3 on the cited page, OR
      - is the project/dataset/model/feature name that the cited resource describes, OR
      - accurately describes what is being cited from that resource (and the source is
        clearly the cited entity per URL/domain/author).
    Reject only when the cited title has no meaningful presence on the resource —
    e.g. a fabricated heading that simply doesn't appear on the page.

==== AUTHORS: LENIENT ====
Be tolerant of author-list mismatches that arise from how academic results display authors.
The following are NOT mismatches:
- Author order differs from the citation. Order doesn't matter — what matters is the set of authors.
- Initials in the result vs full given names in the citation, OR vice versa, when surnames
  and initial(s) are consistent (e.g. "JN Mazón" ↔ "Juan-Nicolás Mazón" is fine; "AS Kumar"
  ↔ "A. Senthil Kumar" is fine).
- Truncated author lists: the citation lists only first N authors or uses "et al." while the
  result lists all authors (or vice versa). Accept if the surnames present in BOTH the
  citation AND the result agree (modulo initial/full-name conventions above).
- Foreign-name transliteration variants and missing diacritics in either side.
- Hyphens / spaces in compound surnames.

Reject on AUTHORS only when:
- There is no overlap in surnames between the citation's author list and the result's authors.
- A non-trivial subset of cited authors is clearly absent from the result and the result lists
  totally different surnames in their place.

==== VENUE & YEAR ====
If the title matches strictly, accept venue/year variations:
- arXiv preprint ↔ published journal/conference version of the SAME EXACT title.
- Conference acronym vs full name (NeurIPS / NIPS / ICLR / ACL …).
- Year differences within ±2 years for arXiv ↔ official publication of the SAME EXACT title.
- Journal name spelling/abbreviation variants for the SAME EXACT title.

==== EVIDENCE QUALITY ====
Acceptable evidence: publisher pages, DOI pages, arXiv abstract, ACL Anthology, PubMed,
NeurIPS/ICML proceedings pages, ResearchGate, Semantic Scholar, official institutional pages.

Reject when the only "evidence" is a page that contains the citation text inside an unrelated
paper's bibliography with no independent publication record.

==== DECISION ====
Set match=true only when at least one search result has a title that EXACTLY matches the
citation title (subject to the trivial-difference list above) AND the authors agree under
the lenient rules above.

Return ONLY JSON, no markdown:
{{"match": true|false, "matched_result": <result_number_or_null>, "note": "<short reason>"}}"""

def parse_json_from_text(text: str) -> Dict:
    """Extract the first JSON object from a model response."""
    if not text:
        return {}
    start = text.find('{')
    end = text.rfind('}') + 1
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}

def _gemini_generate(prompt: str, max_retries: int = 3) -> str:
    # max_retries = total attempts (initial + retries). Default 3 = 1 initial + 2 retries.
    """Call Gemini for structured extraction and matching decisions."""
    if not GEMINI_API_KEY or genai is None:
        return ""

    client = genai.Client(api_key=GEMINI_API_KEY)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text or ""
        except Exception as e:
            last_err = e
            err = str(e).lower()
            if any(p in err for p in _GEMINI_FATAL_PATTERNS):
                raise GeminiFatalError(f"Gemini auth/permission error: {e}")
            if "429" in err or "quota" in err or "resource_exhausted" in err:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 3
                    print(f"    Gemini rate limit/error, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                raise GeminiFatalError(
                    f"Gemini quota/rate limit persisted after {max_retries} retries: {e}"
                )
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    Gemini error: {e}")
            return ""
    if last_err is not None:
        print(f"    Gemini error: {last_err}")
    return ""

def _openai_generate_json(prompt: str, client: Optional[openai.OpenAI]) -> Dict:
    """Fallback JSON judgment with the caller's OpenAI-compatible client."""
    if client is None:
        return {}
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        content = response.choices[0].message.content
        print(f"  LLM Response: {content}")
        return parse_json_from_text(content)
    except Exception as e:
        print(f"  LLM error: {e}")
        return {}

def _judge_with_gemini(prompt: str, client: Optional[openai.OpenAI] = None) -> Dict:
    """Prefer Gemini for JSON comparison; fall back to the provided OpenAI client."""
    text = _gemini_generate(prompt)
    if text:
        print(f"  Gemini Response: {text}")
        result = parse_json_from_text(text)
        if result:
            return result
    return _openai_generate_json(prompt, client)

def extract_metadata_with_gemini(ref: Dict, client: Optional[openai.OpenAI] = None) -> Dict:
    """Normalize/extract citation metadata before searching."""
    raw = ref.get("raw") or "; ".join(
        part for part in [
            ref.get("authors", ""),
            ref.get("title", ""),
            ref.get("venue", ""),
            str(ref.get("year", "")),
        ] if part and part != "N/A"
    )

    if not raw.strip():
        return ref

    prompt = EXTRACT_PROMPT.format(raw=raw)
    text = _gemini_generate(prompt)
    extracted = parse_json_from_text(text)
    if not extracted and client is not None:
        extracted = _openai_generate_json(prompt, client)
    if not extracted:
        return ref

    normalized = dict(ref)
    for key in ("title", "authors", "venue", "year"):
        value = extracted.get(key)
        if value and value != "N/A":
            normalized[key] = value
        elif not normalized.get(key):
            normalized[key] = "N/A"
    normalized["raw"] = ref.get("raw", raw)
    normalized["metadata_method"] = "gemini_extract"
    return normalized

def search_google(query: str, max_retries: int = 3) -> List[Dict]:
    """Search Google via SerpApi Google Search Engine Results API."""
    return _search_google_serpapi(query, max_retries=max_retries)


def _search_google_serpapi(query: str, max_retries: int = 3) -> List[Dict]:
    """Search Google via SerpApi Google Search Engine Results API."""
    client = serpapi.Client(api_key=SERPAPI_KEY)
    params = {
        "engine": "google",
        "q": query,
        "google_domain": "google.com",
        "hl": "en",
        "gl": "us",
        "device": "desktop",
        "num": 10,
    }

    for attempt in range(max_retries):
        try:
            data = client.search(params)
            if data.get("error"):
                err_msg = str(data.get("error", "")).lower()
                if any(p in err_msg for p in _SERPAPI_FATAL_PATTERNS):
                    raise SerpAPIFatalError(f"SerpApi Google Search fatal: {data.get('error')}")
                print(f"  SerpApi Google Search error: {data.get('error')}")
                return []

            items = data.get("organic_results", [])
            return [
                {
                    "position": item.get("position"),
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "displayed_link": item.get("displayed_link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": item.get("source", ""),
                }
                for item in items
                if item.get("link")
            ][:10]
        except SerpAPIFatalError:
            raise
        except Exception as e:
            err = str(e).lower()
            if any(p in err for p in _SERPAPI_FATAL_PATTERNS):
                raise SerpAPIFatalError(f"SerpApi Google Search fatal: {e}")
            if ("429" in err or "rate" in err or "timeout" in err or "temporarily" in err) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 2
                print(f"    SerpApi Google Search error, waiting {wait_time}s before retry {attempt+1}/{max_retries}: {e}")
                time.sleep(wait_time)
                continue
            if "429" in err and attempt >= max_retries - 1:
                raise SerpAPIFatalError(
                    f"SerpApi Google Search rate-limited after {max_retries} retries: {e}"
                )
            print(f"  SerpApi Google Search error: {e}")
            return []
    return []

def search_google_scholar_serpapi(query: str, max_retries: int = 3) -> List[Dict]:
    """Search Google Scholar via SerpApi Python SDK, return top 10 results."""
    global SKIP_SERPAPI

    # If SerpAPI quota is exhausted, skip immediately
    if SKIP_SERPAPI:
        print(f"  Skipping SerpAPI (quota exhausted)")
        return []

    client = serpapi.Client(api_key=SERPAPI_KEY)
    params = {
        "engine": "google_scholar",
        "q": query,
        "hl": "en",
        "num": 10
    }

    for attempt in range(max_retries):
        try:
            results = client.search(params)
            if results.get("error"):
                err_msg = str(results.get("error", "")).lower()
                if any(p in err_msg for p in _SERPAPI_FATAL_PATTERNS):
                    raise SerpAPIFatalError(f"SerpApi Scholar fatal: {results.get('error')}")
                print(f"  SerpAPI error: {results.get('error')}")
                return []

            organic_results = results.get("organic_results", [])

            formatted_results = []
            for result in organic_results:
                pub_info = result.get("publication_info", {})
                authors_list = pub_info.get("authors", [])
                author_names = [a.get("name", "") for a in authors_list] if authors_list else []

                formatted_results.append({
                    "title": result.get("title", ""),
                    "authors": ", ".join(author_names),
                    "snippet": result.get("snippet", ""),
                    "publication_info": pub_info.get("summary", ""),
                    "link": result.get("link", "")
                })

            return formatted_results
        except SerpAPIFatalError:
            raise
        except Exception as e:
            err = str(e).lower()
            if any(p in err for p in _SERPAPI_FATAL_PATTERNS):
                raise SerpAPIFatalError(f"SerpApi Scholar fatal: {e}")
            if "429" in err and attempt >= max_retries - 1:
                raise SerpAPIFatalError(
                    f"SerpApi Scholar rate-limited after {max_retries} retries: {e}"
                )
            if ("429" in err or "rate" in err or "timeout" in err or "temporarily" in err) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 3
                print(f"    SerpAPI error: {e}, waiting {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            print(f"  SerpAPI error: {e}")
            return []
    return []

def download_content(url: str) -> str:
    """Download URL content (PDF or webpage text) - get more content for better matching"""
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get('Content-Type', '').lower()

        if 'pdf' in content_type or url.endswith('.pdf'):
            # Save PDF and extract text from first 3 pages
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                f.write(resp.content)
                temp_path = f.name
            try:
                # Extract first 3 pages for better matching
                text_parts = []
                for page_num in range(min(3, 10)):  # Try up to 3 pages
                    try:
                        page_text = get_page_text(temp_path, page_num)
                        text_parts.append(page_text)
                    except:
                        break
                text = '\n'.join(text_parts)[:5000]  # Take first 5000 chars
                return text
            except:
                return None
        else:
            # Use BeautifulSoup for HTML to avoid Playwright issues
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text[:5000]
    except Exception as e:
        print(f"    Download error for {url}: {e}")
        return None

def verify_with_serp(ref: Dict, client: openai.OpenAI) -> Dict:
    """Verify citation with three-step process:
    0. Gemini extracts/normalizes structured metadata, then direct URL check if present
    1. SerpApi Google Search API + downloaded/snippet content + Gemini strict match
    2. SerpApi Google Scholar API + Gemini strict match
    3. If still failed, return not found
    """
    global thread_error_pause

    # Check if we need to pause due to thread exhaustion
    with thread_error_lock:
        if thread_error_pause:
            print(f"  ⚠ Thread exhaustion detected, pausing for 5 seconds...")
            time.sleep(5)
            thread_error_pause = False

    try:
        return _verify_with_serp_impl(ref, client)
    except RuntimeError as e:
        if "can't start new thread" in str(e):
            with thread_error_lock:
                thread_error_pause = True
                print(f"  ⚠ Thread exhaustion error, triggering 5-second pause for all workers...")
            # Wait and retry once
            time.sleep(5)
            return _verify_with_serp_impl(ref, client)
        else:
            raise

def _verify_with_serp_impl(ref: Dict, client: openai.OpenAI) -> Dict:
    """Internal implementation of verify_with_serp"""
    global thread_error_pause

    print(f"  Step 0: Gemini metadata extraction...")
    ref = extract_metadata_with_gemini(ref, client)

    title = ref.get('title', '')
    authors = ref.get('authors', '')
    venue = ref.get('venue', 'N/A')
    year = ref.get('year', 'N/A')
    raw = ref.get('raw', '')

    if not title:
        return {
            "found": False,
            "method": "metadata_extraction_failed",
            "error": "missing_title",
            "citation": ref,
            "reason": "Could not extract a title before searching"
        }

    # Step 0: Check if URL is provided in the citation
    import re

    # First, clean up common OCR errors in URLs (spaces in numbers)
    # e.g., "https://arxiv.org/abs/2412. 08905" -> "https://arxiv.org/abs/2412.08905"
    cleaned_raw = re.sub(r'(https?://[^\s]+)\.\s+(\d+)', r'\1.\2', raw)

    url_patterns = [
        r'https?://[^\s,)\]]+',  # Standard URLs (stop at space, comma, closing paren/bracket)
        r'arxiv:\s*(\d+\.\d+)',  # arXiv format like "arxiv:2412.08905"
        r'arXiv:\s*(\d+\.\d+)',  # arXiv with capital X
    ]

    found_url = None
    for pattern in url_patterns:
        match = re.search(pattern, cleaned_raw, re.IGNORECASE)
        if match:
            if 'arxiv' in pattern.lower():
                # Convert arxiv:XXXX to URL
                arxiv_id = match.group(1)
                found_url = f"https://arxiv.org/abs/{arxiv_id}"
            else:
                found_url = match.group(0)
                # Remove trailing punctuation (., ,, ;, etc.) but keep valid URL chars
                found_url = re.sub(r'[.,;:!?]+$', '', found_url)
            break

    if found_url:
        print(f"  Step 0: Found URL in citation: {found_url} — trying it first")
        try:
            resp = requests.get(
                found_url, timeout=20,
                headers={"User-Agent": "Mozilla/5.0"}, allow_redirects=True,
            )
            if resp.status_code < 400:
                content = download_content(found_url)
                if content:
                    prompt = COMPARE_PROMPT.format(
                        title=title, authors=authors, venue=venue, year=year,
                        documents=f"Result 1 (from provided URL {found_url}):\n{content}\n",
                    )
                    try:
                        result = _judge_with_gemini(prompt, client)
                        if result.get('match', False):
                            result['method'] = 'direct_url'
                            result['found'] = True
                            result['reason'] = result.get('note', 'Verified via provided URL')
                            result['citation'] = ref
                            print(f"  ✓ Verified via provided URL")
                            return result
                        print(f"  ✗ URL content didn't match — falling through to search")
                    except Exception as e:
                        print(f"  ⚠ URL verification error: {e} — falling through to search")
                else:
                    print(f"  ⚠ URL content unreadable — falling through to search")
            else:
                # URLs change/expire; treat any HTTP failure as "URL stale", not "citation fake".
                print(f"  ⚠ URL returned HTTP {resp.status_code} — falling through to search")
        except (requests.Timeout, requests.ConnectionError) as e:
            print(f"  ⚠ URL access error: {e} — falling through to search")
        except Exception as e:
            print(f"  ⚠ URL access unexpected error: {e} — falling through to search")
        # Fall through to Step 1 + Step 2 — title/authors search may still find the paper.

    query = f'"{title}"'
    if authors:
        first_author = authors.split(",")[0].split(" and ")[0].strip()
        query += f" {first_author}"

    # Step 1: Try SerpApi Google Search API
    print(f"  Step 1: SerpApi Google Search API...")
    results = search_google(query)
    if results:
        urls = [r.get('link') for r in results if r.get('link')][:5]

        # Download all URLs in parallel with error handling
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                contents = list(executor.map(download_content, urls))
        except RuntimeError as e:
            if "can't start new thread" in str(e):
                with thread_error_lock:
                    global thread_error_pause
                    thread_error_pause = True
                    print(f"  ⚠ Thread exhaustion in download, pausing 5 seconds...")
                time.sleep(5)
                # Retry with parallel download
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    contents = list(executor.map(download_content, urls))
            else:
                raise

        # Format documents for GPT
        docs_text = []
        for i, (url, content) in enumerate(zip(urls, contents)):
            result_meta = results[i] if i < len(results) else {}
            if content:
                docs_text.append(
                    f"Result {i+1} (from {url}):\n"
                    f"Title: {result_meta.get('title', 'N/A')}\n"
                    f"Snippet: {result_meta.get('snippet', 'N/A')}\n"
                    f"Downloaded content:\n{content}\n"
                )
            else:
                docs_text.append(
                    f"Result {i+1} (from {url}):\n"
                    f"Title: {result_meta.get('title', 'N/A')}\n"
                    f"Snippet: {result_meta.get('snippet', 'N/A')}\n"
                    "[Failed to download]\n"
                )

        if docs_text:
            prompt = COMPARE_PROMPT.format(
                title=title, authors=authors,
                venue=venue, year=year,
                documents="\n".join(docs_text)
            )

            try:
                print(f"  Sending to Gemini for Step 1 verification...")
                result = _judge_with_gemini(prompt, client)

                if result.get('match', False):
                    result['method'] = 'serpapi_google_search'
                    result['found'] = True
                    result['reason'] = result.get('note', 'Verified via SerpApi Google Search')
                    result['citation'] = ref
                    print(f"  ✓ Found via SerpApi Google Search")
                    return result
                else:
                    print(f"  ✗ Step 1 LLM said no match: {result.get('note', 'N/A')}")
            except Exception as e:
                print(f"  SerpApi Google Search LLM error: {e}")

    # Step 2: Try SerpAPI Google Scholar
    print(f"  Step 2: SerpAPI Google Scholar...")
    scholar_results = search_google_scholar_serpapi(query)

    if scholar_results:
        # Format results for GPT
        docs_text = []
        for i, result in enumerate(scholar_results):
            docs_text.append(f"""Result {i+1}:
Title: {result.get('title', 'N/A')}
Authors: {result.get('authors', 'N/A')}
Publication Info: {result.get('publication_info', 'N/A')}
Snippet: {result.get('snippet', 'N/A')}
Link: {result.get('link', 'N/A')}
""")

        prompt = COMPARE_PROMPT.format(
            title=title, authors=authors,
            venue=venue, year=year,
            documents="\n".join(docs_text)
        )

        try:
            print(f"  Sending to Gemini for Step 2 verification...")
            result = _judge_with_gemini(prompt, client)

            if result.get('match', False):
                result['method'] = 'serpapi_google_scholar'
                result['found'] = True
                result['reason'] = result.get('note', 'Verified via SerpAPI Google Scholar')
                result['citation'] = ref
                print(f"  ✓ Found via SerpAPI Google Scholar")
                return result
            else:
                print(f"  ✗ Step 2 LLM said no match: {result.get('note', 'N/A')}")
        except Exception as e:
            print(f"  SerpAPI Google Scholar LLM error: {e}")

    # Step 3: Not found in either method
    return {
        "found": False,
        "method": "both_failed",
        "error": "not_found_in_both_methods",
        "citation": ref,
        "reason": "Citation not found via SerpApi Google Search or SerpAPI Google Scholar - title or authors may not match any existing publications"
    }
