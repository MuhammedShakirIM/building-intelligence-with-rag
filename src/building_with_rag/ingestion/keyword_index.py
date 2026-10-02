"""Create or reuse the Atlas Search keyword index on chunks.text (Story 4.1).

Run: uv run python -m building_with_rag.ingestion.keyword_index
No Voyage calls, no data writes. A differing existing index is reported, never replaced.
"""

import sys
import time

from pymongo.operations import SearchIndexModel

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.retrieval.semantic import _db
from building_with_rag.settings import get_settings

WAIT_SECONDS = 120
POLL_SECONDS = 3


def _definition(index: dict) -> dict:
    return index.get("latestDefinition") or index.get("definition") or {}


def _diff(expected: dict, actual: dict, path: str = "") -> list[str]:
    out = []
    for key in sorted(set(expected) | set(actual)):
        p = f"{path}.{key}" if path else key
        if key not in actual:
            out.append(f"{p}: missing")
        elif key not in expected:
            out.append(f"{p}: unexpected")
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            out.extend(_diff(expected[key], actual[key], p))
        elif expected[key] != actual[key]:
            out.append(f"{p}: expected {expected[key]!r}, found {actual[key]!r}")
    return out


def main() -> int:
    settings = get_settings()
    if not settings.mongodb_uri:
        print("MONGODB_URI is not set in .env.")
        return 1
    chunks = _db(settings)[schema.CHUNKS_COLLECTION]
    name = schema.KEYWORD_INDEX_NAME
    existing = list(chunks.list_search_indexes(name))
    if existing:
        differences = _diff(schema.KEYWORD_INDEX_DEFINITION, _definition(existing[0]))
        if differences:
            print(f"{name} exists with a different definition; drop it manually in Atlas:")
            for line in differences:
                print(f"- {line}")
            return 1
        print(f"{name}: reusing existing index")
    else:
        chunks.create_search_index(SearchIndexModel(
            name=name, type="search", definition=schema.KEYWORD_INDEX_DEFINITION))
        print(f"{name}: created")
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        current = next(iter(chunks.list_search_indexes(name)), {})
        if current.get("queryable") or current.get("status") == "READY":
            print(f"{name}: {current.get('status', 'READY')} (queryable)")
            return 0
        if time.monotonic() >= deadline:
            print(f"{name}: not READY after {WAIT_SECONDS}s (status {current.get('status')})")
            return 1
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
