from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from agent.preprocess import sanitize_text
from agent.schemas import Document, IndexNode, Page

HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节条部分]|"
    r"[一二三四五六七八九十]+[、.]|[0-9]+(?:\.[0-9]+){0,3}[、.\s]|"
    r"[A-Z][、.\s])"
)


class PageIndexStore:
    def __init__(self, root: Path):
        self.root = root

    def document_dir(self, document: Document | str) -> Path:
        if isinstance(document, Document):
            return self.root / document.domain / document.doc_id
        matches = list(self.root.glob(f"*/{document}"))
        if not matches:
            raise FileNotFoundError(f"No processed index for {document}")
        return matches[0]

    def save(self, document: Document, pages: list[Page], root: IndexNode) -> None:
        output_dir = self.document_dir(document)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "doc_id": document.doc_id,
            "domain": document.domain,
            "title": document.title,
            "source_path": str(document.path),
            "page_count": len(pages),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "page_index.json").write_text(
            json.dumps(root.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_dir / "pages.jsonl").open("w", encoding="utf-8") as handle:
            for page in pages:
                handle.write(json.dumps(
                    {"page_number": page.page_number, "text": sanitize_text(page.text)},
                    ensure_ascii=False,
                ))
                handle.write("\n")

    def load_index(self, doc_id: str) -> IndexNode:
        path = self.document_dir(doc_id) / "page_index.json"
        return IndexNode.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def load_pages(self, doc_id: str, start_page: int, end_page: int) -> list[Page]:
        path = self.document_dir(doc_id) / "pages.jsonl"
        pages: list[Page] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                number = int(item["page_number"])
                if start_page <= number <= end_page:
                    pages.append(Page(page_number=number, text=item["text"]))
        return pages


def build_page_index(
    document: Document,
    pages: list[Page],
    leaf_pages: int = 6,
    branch_factor: int = 8,
) -> IndexNode:
    leaves: list[IndexNode] = []
    for start in range(0, len(pages), leaf_pages):
        group = pages[start:start + leaf_pages]
        leaves.append(
            IndexNode(
                node_id=f"p{group[0].page_number}-{group[-1].page_number}",
                title=_group_title(group),
                start_page=group[0].page_number,
                end_page=group[-1].page_number,
            )
        )
    level = leaves
    depth = 1
    while len(level) > branch_factor:
        parents: list[IndexNode] = []
        for start in range(0, len(level), branch_factor):
            children = level[start:start + branch_factor]
            parents.append(
                IndexNode(
                    node_id=f"l{depth}-{children[0].start_page}-{children[-1].end_page}",
                    title=_parent_title(children),
                    start_page=children[0].start_page,
                    end_page=children[-1].end_page,
                    children=children,
                )
            )
        level = parents
        depth += 1
    content_title = _group_title(pages[: min(len(pages), 2)])
    root_title = document.title
    if content_title and content_title.lower() not in document.title.lower():
        root_title = f"{document.title} | {content_title}"
    return IndexNode(
        node_id="root",
        title=root_title,
        start_page=1,
        end_page=max(1, len(pages)),
        children=level,
    )


def flatten_nodes(root: IndexNode, leaves_only: bool = False) -> list[IndexNode]:
    output: list[IndexNode] = []

    def visit(node: IndexNode) -> None:
        if not leaves_only or not node.children:
            output.append(node)
        for child in node.children:
            visit(child)

    visit(root)
    return output


def compact_tree(root: IndexNode) -> str:
    lines: list[str] = []

    def visit(nodes: Iterable[IndexNode], depth: int) -> None:
        for node in nodes:
            lines.append(
                f"{'  ' * depth}- {node.node_id} | pages {node.start_page}-{node.end_page} | {node.title}"
            )
            visit(node.children, depth + 1)

    visit(root.children, 0)
    return "\n".join(lines)


def _group_title(pages: list[Page]) -> str:
    candidates: list[str] = []
    for page in pages:
        for line in page.text.splitlines()[:30]:
            line = line.strip()
            if 4 <= len(line) <= 100 and (HEADING_RE.match(line) or len(candidates) == 0):
                candidates.append(line)
                if len(candidates) == 2:
                    return " / ".join(candidates)
    return candidates[0] if candidates else f"Pages {pages[0].page_number}-{pages[-1].page_number}"


def _parent_title(children: list[IndexNode]) -> str:
    first = children[0].title
    last = children[-1].title
    return first if first == last else f"{first} ... {last}"
