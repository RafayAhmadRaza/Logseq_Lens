"""
Enhanced Logseq parsing utilities.
Extracts links, tags, properties, and other graph features from parsed Logseq pages.
"""
import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import LogseqMarkdownParser


@dataclass
class ParsedPage:
    """Enhanced parsed page with extracted graph features."""
    source: str
    page_properties: Dict[str, str]
    page_content: str
    blocks: List[Dict]
    # Extracted features
    page_links: List[str] = field(default_factory=list)  # [[Page Name]]
    tags: List[str] = field(default_factory=list)        # #tag
    block_uuids: List[str] = field(default_factory=list)
    todo_items: List[Dict] = field(default_factory=list)
    journal_date: Optional[str] = None


# Regex patterns
PAGE_LINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')
TAG_PATTERN = re.compile(r'(?:^|\s)#([\w\-/]+)')
JOURNAL_DATE_PATTERN = re.compile(r'journals?/(\d{4}_\d{2}_\d{2})\.md$', re.IGNORECASE)


def extract_page_links(text: str) -> List[str]:
    """Extract all [[Page Name]] links from text."""
    return PAGE_LINK_PATTERN.findall(text)


def extract_tags(text: str) -> List[str]:
    """Extract all #tags from text."""
    return TAG_PATTERN.findall(text)


def extract_journal_date(source_path: str) -> Optional[str]:
    """Extract date from journal page path."""
    match = JOURNAL_DATE_PATTERN.search(source_path)
    if match:
        return match.group(1).replace('_', '-')
    return None


def parse_logseq_page_enhanced(raw_doc: Dict) -> ParsedPage:
    """
    Parse a raw Logseq document into an enhanced ParsedPage with graph features.
    
    Args:
        raw_doc: Dict with 'content' and 'source' keys
        
    Returns:
        ParsedPage with extracted links, tags, properties, etc.
    """
    page = LogseqMarkdownParser.parse_text(raw_doc["content"])
    parsed = page.dict()
    
    # Extract links from all blocks
    all_links = []
    all_tags = []
    all_uuids = []
    todo_items = []
    
    for block in parsed.get("blocks", []):
        block_content = block.get("block_content", "")
        all_links.extend(extract_page_links(block_content))
        all_tags.extend(extract_tags(block_content))
        
        # UUID
        uuid = block.get("block_UUID")
        if uuid:
            all_uuids.append(uuid)
        
        # TODO items
        todo_state = block.get("block_TODO_state")
        if todo_state:
            todo_items.append({
                "uuid": uuid,
                "state": todo_state,
                "content": block_content[:200]
            })
    
    # Journal date from path
    journal_date = extract_journal_date(raw_doc["source"])
    
    return ParsedPage(
        source=raw_doc["source"],
        page_properties=parsed.get("page_properties", {}),
        page_content=parsed.get("page_content", ""),
        blocks=parsed.get("blocks", []),
        page_links=list(set(all_links)),  # Deduplicate
        tags=list(set(all_tags)),
        block_uuids=all_uuids,
        todo_items=todo_items,
        journal_date=journal_date
    )


def parse_documents_enhanced(docs: List[Dict]) -> List[ParsedPage]:
    """Parse multiple raw documents into enhanced ParsedPages."""
    return [parse_logseq_page_enhanced(doc) for doc in docs]


def get_all_page_titles(parsed_pages: List[ParsedPage]) -> Set[str]:
    """Get all page titles from parsed pages (filenames without extension)."""
    from pathlib import Path
    titles = set()
    for page in parsed_pages:
        title = Path(page.source).stem
        titles.add(title)
        # Also add from page properties if there's a title property
        if "title" in page.page_properties:
            titles.add(page.page_properties["title"])
    return titles


def find_pages_by_title(parsed_pages: List[ParsedPage], title: str) -> List[ParsedPage]:
    """Find pages matching a title (exact or partial)."""
    title_lower = title.lower()
    results = []
    for page in parsed_pages:
        page_title = Path(page.source).stem.lower()
        if title_lower == page_title or title_lower in page_title:
            results.append(page)
        # Also check page properties
        if "title" in page.page_properties:
            if title_lower == page.page_properties["title"].lower() or title_lower in page.page_properties["title"].lower():
                if page not in results:
                    results.append(page)
    return results


def find_pages_by_tag(parsed_pages: List[ParsedPage], tag: str) -> List[ParsedPage]:
    """Find pages containing a specific tag."""
    tag_lower = tag.lower().lstrip('#')
    results = []
    for page in parsed_pages:
        for page_tag in page.tags:
            if tag_lower == page_tag.lower() or tag_lower in page_tag.lower():
                results.append(page)
                break
    return results


def find_related_pages(parsed_pages: List[ParsedPage], page: ParsedPage, max_results: int = 10) -> List[ParsedPage]:
    """
    Find pages related to the given page via:
    - Direct links ([[Page Name]])
    - Shared tags
    - Backlinks (pages that link to this page)
    """
    # Build title -> page map
    title_to_page = {}
    for p in parsed_pages:
        title = Path(p.source).stem
        title_to_page[title.lower()] = p
        if "title" in p.page_properties:
            title_to_page[p.page_properties["title"].lower()] = p
    
    related = []
    seen = {page.source}
    
    # 1. Forward links
    for link in page.page_links:
        link_lower = link.lower()
        if link_lower in title_to_page:
            linked_page = title_to_page[link_lower]
            if linked_page.source not in seen:
                related.append(linked_page)
                seen.add(linked_page.source)
    
    # 2. Backlinks (pages that link to this page)
    page_title = Path(page.source).stem
    for p in parsed_pages:
        if p.source in seen:
            continue
        if page_title.lower() in [l.lower() for l in p.page_links]:
            related.append(p)
            seen.add(p.source)
    
    # 3. Shared tags
    page_tags = set(t.lower() for t in page.tags)
    if page_tags:
        for p in parsed_pages:
            if p.source in seen:
                continue
            p_tags = set(t.lower() for t in p.tags)
            if page_tags & p_tags:  # Intersection
                related.append(p)
                seen.add(p.source)
    
    return related[:max_results]


def build_page_index(parsed_pages: List[ParsedPage]) -> Dict:
    """Build an index for fast lookups."""
    from pathlib import Path
    index = {
        "by_source": {p.source: p for p in parsed_pages},
        "by_title": {},
        "by_tag": {},
        "by_uuid": {},
    }
    
    for page in parsed_pages:
        title = Path(page.source).stem
        index["by_title"][title.lower()] = page
        if "title" in page.page_properties:
            index["by_title"][page.page_properties["title"].lower()] = page
        
        for tag in page.tags:
            tag_lower = tag.lower()
            if tag_lower not in index["by_tag"]:
                index["by_tag"][tag_lower] = []
            index["by_tag"][tag_lower].append(page)
        
        for uuid in page.block_uuids:
            index["by_uuid"][uuid] = page
    
    return index