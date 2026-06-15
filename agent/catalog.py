from __future__ import annotations

import json
from pathlib import Path

from agent.schemas import Document, Question

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".html", ".htm"}
SOURCE_PRIORITY = {".txt": 0, ".html": 1, ".htm": 1, ".pdf": 2}


class DatasetCatalog:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.documents = self._discover_documents()

    def _discover_documents(self) -> dict[str, Document]:
        documents: dict[str, Document] = {}
        raw_root = self.dataset_root / "raw"
        for domain_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            candidates = sorted(
                (
                    path
                    for path in domain_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
                ),
                key=lambda path: (path.stem, SOURCE_PRIORITY[path.suffix.lower()]),
            )
            for path in candidates:
                document = Document(
                    doc_id=path.stem,
                    domain=domain_dir.name,
                    path=path,
                    title=_title_from_stem(path.stem),
                )
                current = documents.get(document.doc_id)
                if current is None or SOURCE_PRIORITY[path.suffix.lower()] < SOURCE_PRIORITY[current.path.suffix.lower()]:
                    documents[document.doc_id] = document
        return documents

    def get_document(self, doc_id: str) -> Document:
        try:
            return self.documents[doc_id]
        except KeyError as exc:
            raise KeyError(f"Unknown doc_id: {doc_id}") from exc

    def domain_documents(self, domain: str) -> list[Document]:
        return [doc for doc in self.documents.values() if doc.domain == domain]

    def load_questions(self, path: Path) -> list[Question]:
        files = sorted(path.glob("*.json")) if path.is_dir() else [path]
        questions: list[Question] = []
        for file_path in files:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            questions.extend(Question.from_dict(item) for item in payload)
        return questions

    def validate_questions(self, questions: list[Question]) -> list[str]:
        return sorted(
            {
                doc_id
                for question in questions
                for doc_id in question.doc_ids
                if doc_id not in self.documents
            }
        )


def _title_from_stem(stem: str) -> str:
    title = stem.replace("_", " ").strip()
    return title if len(title) <= 120 else title[:117] + "..."

