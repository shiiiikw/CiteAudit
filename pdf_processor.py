"""PDF processing: detect Reference section using GPT, extract text"""
import fitz  # PyMuPDF
import openai
import json
from typing import Tuple, List

FIND_REF_PROMPT = """Analyze these pages and identify which ones contain References/Bibliography.

{page_summaries}

Return JSON: {{"citation_pages": [list of page numbers that contain references]}}
- Only include pages that actually contain reference entries (numbered citations like [1], [2] or author-year format)
- Do NOT include appendix, supplementary materials, or main content pages
- Page numbers are 1-indexed"""

def get_page_summary(pdf_path: str, page_num: int) -> str:
    """Get first 500 + last 500 chars of a page"""
    doc = fitz.open(pdf_path)
    text = doc[page_num].get_text()
    doc.close()
    if len(text) <= 1000:
        return text.replace('\n', ' ')
    return (text[:500] + " ... " + text[-500:]).replace('\n', ' ')

def find_reference_section(pdf_path: str, client: openai.OpenAI) -> Tuple[int, int]:
    """Use GPT to find Reference section page range (0-indexed), batch 20 pages at a time"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    all_citation_pages = []
    batch_size = 20

    for batch_start in range(0, total_pages, batch_size):
        batch_end = min(batch_start + batch_size, total_pages)
        summaries = []
        for i in range(batch_start, batch_end):
            text = get_page_summary(pdf_path, i)
            summaries.append(f"Page {i+1}: {text}")

        prompt = FIND_REF_PROMPT.format(page_summaries="\n\n".join(summaries))

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200
        )

        content = response.choices[0].message.content
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            result = json.loads(content[start:end])
            pages = result.get('citation_pages', [])
            all_citation_pages.extend(pages)
        except:
            pass

    if not all_citation_pages:
        return -1, -1

    # Convert to 0-indexed, return range
    ref_start = min(all_citation_pages) - 1
    ref_end = max(all_citation_pages)  # exclusive
    return ref_start, ref_end

def get_page_text(pdf_path: str, page_num: int) -> str:
    """Extract text from a single PDF page"""
    doc = fitz.open(pdf_path)
    text = doc[page_num].get_text()
    doc.close()
    return text

def get_total_pages(pdf_path: str) -> int:
    """Get total page count"""
    doc = fitz.open(pdf_path)
    total = len(doc)
    doc.close()
    return total
