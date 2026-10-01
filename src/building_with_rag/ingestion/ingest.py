"""Ingestion runner: corpus MongoDB, chunk, embed, index.

Run with::

    uv run python -m building_with_rag.ingestion.ingest
"""

import json
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from voyageai import Client as VoyageClient
from voyageai.error import VoyageError

from building_with_rag.ingestion import mongodb_schema as schema
from building_with_rag.settings import get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/building_with_rag/ingestion *
BNS_CORPUS = _PROJECT_ROOT / "data" / "processed" / "bns_sections.jsonl"
IPC_CORPUS = _PROJECT_ROOT / "data" / "processed" / "ipc_sections.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, returning a list of dicts with a ``_record_index`` field."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_record_index"] = idx
            records.append(record)
    return records

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def _check_mongodb(settings) -> MongoClient:
    """Run MongoDB prerequisite checks; return a connected client or raise SystemExit."""
    uri = settings.mongodb_uri
    if not uri:
        raise SystemExit("MongoDB unavailable: MONGODB_URI is empty. Set it in .env and re-run.")

    try:
        client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
    except PyMongoError as exc:
        raise SystemExit(
            f"MongoDB unavailable: could not connect to {uri}. "
            f"Check MONGODB_URI and that the deployment is running. ({exc})"
        ) from exc

    # Detect deployment type and vector-search capability
    try:
        build_info = client.admin.command("buildInfo")
        version_str: str = build_info.get("version", "")
        major = int(version_str.split(".")[0]) if version_str else 0
    except OperationFailure:
        major = 0

    # Check for Atlas
    try:
        cmd_line = client.admin.command("getCmdLineOpts")
        argv = cmd_line.get("argv", [])
        is_atlas = any("mongodb.net" in arg or "atlas" in arg.lower() for arg in argv)
    except OperationFailure:
        is_atlas = False

    # Also check server status for atlasVersion
    try:
        server_status = client.admin.command("serverStatus")
        atlas_version = server_status.get("atlasVersion")
        if atlas_version:
            is_atlas = True
    except OperationFailure:
        pass

    if not is_atlas and major >= 7:
        # MongoDB 7.0+ community — check if vector search available
        try:
            client.admin.command({"vectorSearch": 1})
        except OperationFailure as exc:
            if "no such command" in str(exc).lower() or "unrecognized" in str(exc).lower():
                raise SystemExit(
                    "MongoDB Community Edition does not support vector search indexes. "
                    "Use MongoDB Atlas (M0 free tier or higher) or MongoDB Enterprise "
                    "with Atlas Search enabled. This project requires $vectorSearch."
                )
    elif not is_atlas and major < 6:
        raise SystemExit(
            f"MongoDB {major}.x does not support vector search. "
            "Use MongoDB Atlas (M0 free tier or higher) or MongoDB Enterprise 6.0+."
        )

    # Check createSearchIndex privilege
    db_name = settings.mongodb_db_name if settings.app_env != "testing" else settings.mongodb_test_db_name
    db = client[db_name]
    try:
        # Test creating and immediately dropping a dummy index on a temp collection
        temp_coll = "__prereq_check_temp__"
        db.create_collection(temp_coll)
        try:
            db.command(
                "createSearchIndexes",
                temp_coll,
                indexes=[{"name": "temp", "type": "vectorSearch", "definition": {}}],
            )
        except OperationFailure:
            pass  # May fail on definition validation; the key is we weren't denied
        finally:
            db.drop_collection(temp_coll)
    except OperationFailure as exc:
        if exc.code == 13 or "not authorized" in str(exc).lower():
            raise SystemExit(
                "Permission missing: the connected user cannot create search indexes. "
                "Grant the atlasAdmin role (Atlas) or equivalent, then re-run."
            ) from exc
        # Other errors (e.g. invalid definition) are fine — means we have permission

    return client

# ---------------------------------------------------------------------------
# Ingestion steps
# ---------------------------------------------------------------------------


def _ingest_sources(db, sources_coll: str, bns_records: list[dict], ipc_records: list[dict]) -> dict:
    """Upsert source documents. Returns {inserted, replaced, skipped}."""
    stats = {"inserted": 0, "replaced": 0, "skipped": 0}
    collection = db[sources_coll]

    datasets = [
        (bns_records, "BNS_2023", "bns_sections.jsonl"),
        (ipc_records, "IPC_1860", "ipc_sections.jsonl"),
    ]

    for records, act, filename in datasets:
        first = records[0]
        doc = schema.build_source_doc(
            act=act,
            act_label=first["act_label"],
            status=first["status"],
            source_pdf=first["source_pdf"],
            source_sha256=first["source_sha256"],
            parser=first["parser"],
            parser_version=first["parser_version"],
            corpus_path=f"data/processed/{filename}",
            section_count=len(records),
        )
        existing = collection.find_one({"_id": doc["_id"]})
        if existing is None:
            collection.insert_one(doc)
            stats["inserted"] += 1
        elif _docs_equal(existing, doc, ("updated_at",)):
            stats["skipped"] += 1
        else:
            doc["created_at"] = existing.get("created_at", doc["created_at"])
            doc["updated_at"] = schema.now_utc()
            collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            stats["replaced"] += 1

    return stats


def _ingest_sections(db, sections_coll: str, records: list[dict]) -> dict:
    """Upsert section documents. Returns {inserted, replaced, skipped}."""
    stats = {"inserted": 0, "replaced": 0, "skipped": 0}
    collection = db[sections_coll]

    for record in records:
        doc = schema.build_section_doc(record)
        existing = collection.find_one({"_id": doc["_id"]})
        if existing is None:
            collection.insert_one(doc)
            stats["inserted"] += 1
        elif _docs_equal(existing, doc, ("updated_at",)):
            stats["skipped"] += 1
        else:
            doc["created_at"] = existing.get("created_at", doc["created_at"])
            doc["updated_at"] = schema.now_utc()
            collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
            stats["replaced"] += 1

    return stats


def _chunk_sections(db, sections_coll: str, chunks_coll: str) -> tuple[int, list[str]]:
    """Chunk all sections; store chunk docs. Returns (new_chunk_count, all_chunk_ids)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=schema.CHUNK_SIZE,
        chunk_overlap=schema.CHUNK_OVERLAP,
        separators=schema.CHUNK_SEPARATORS,
    )
    sections_collection = db[sections_coll]
    chunks_collection = db[chunks_coll]
    new_chunk_ids: list[str] = []
    all_chunk_ids: list[str] = []
    stats = {"inserted": 0, "replaced": 0, "skipped": 0}

    for section in sections_collection.find().sort("section_id", 1):
        text: str = section["text"]
        heading: str = section.get("heading", "")
        act_label: str = section.get("act_label", "")
        if not text.strip():
            continue

        raw_chunks = splitter.split_text(text)
        for idx, chunk_text in enumerate(raw_chunks):
            embedding_input = f"[{act_label}] {heading}\n{chunk_text}".strip()
            doc = schema.build_chunk_doc(
                section=section, chunk_index=idx, text=chunk_text,
                embedding_input=embedding_input,
            )
            all_chunk_ids.append(doc["_id"])

            existing = chunks_collection.find_one({"_id": doc["_id"]})
            if existing is None:
                chunks_collection.insert_one(doc)
                stats["inserted"] += 1
                new_chunk_ids.append(doc["_id"])
            elif existing.get("text") != chunk_text or existing.get("chunking_version") != schema.CHUNKING_VERSION:
                doc["created_at"] = existing.get("created_at", doc["created_at"])
                chunks_collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
                stats["replaced"] += 1
                new_chunk_ids.append(doc["_id"])
            else:
                stats["skipped"] += 1

    # Orphan cleanup: remove chunks whose chunking_version doesn't match
    chunks_collection.delete_many({"chunking_version": {"$ne": schema.CHUNKING_VERSION}})

    # Refresh chunk_ids after cleanup
    all_chunk_ids = [c["_id"] for c in chunks_collection.find({}, {"_id": 1})]

    chunk_count = stats["inserted"] + stats["replaced"]
    print(f"  Chunks: {stats['inserted']} inserted, {stats['replaced']} replaced, "
          f"{stats['skipped']} skipped")
    return chunk_count, all_chunk_ids


VOYAGE_BATCH_SIZE = 10
VOYAGE_BATCH_DELAY = 25  # seconds — 2.4 RPM, ~5K tokens/batch under 10K TPM


def _embed_chunks(db, embeddings_coll: str, chunk_ids: list[str], settings) -> dict:
    """Embed new/changed chunks using Voyage in batches. Returns stats."""
    stats = {"inserted": 0, "replaced": 0, "skipped": 0}
    chunks_collection = db[schema.CHUNKS_COLLECTION]
    embeddings_collection = db[embeddings_coll]

    if not settings.voyage_api_key:
        print("  WARNING: VOYAGE_API_KEY is empty. Skipping embedding step.")
        return stats

    vc = VoyageClient(api_key=settings.voyage_api_key)
    all_chunks = list(chunks_collection.find({"_id": {"$in": chunk_ids}}))

    # Only embed chunks that don't already have a matching embedding
    to_embed: list[dict] = []
    for chunk in all_chunks:
        embedding_id = schema.build_embedding_id(chunk["_id"])
        if embeddings_collection.find_one({"_id": embedding_id}) is None:
            to_embed.append(chunk)
        else:
            stats["skipped"] += 1

    if not to_embed:
        print(f"  Embeddings: 0 inserted, 0 replaced, {stats['skipped']} skipped (all cached)")
        return stats

    # Batch embed
    batches = [
        to_embed[i : i + VOYAGE_BATCH_SIZE]
        for i in range(0, len(to_embed), VOYAGE_BATCH_SIZE)
    ]
    for batch_idx, batch in enumerate(batches):
        texts = [chunk["embedding_input"] for chunk in batch]
        try:
            result = vc.embed(texts=texts, model=schema.EMBEDDING_MODEL, input_type="document")
            vectors = result.embeddings
        except VoyageError as exc:
            # Retry once after a wait on rate-limit
            print(f"  Error on batch {batch_idx + 1}/{len(batches)}: {exc}; "
                  f"waiting 30s then retrying...")
            time.sleep(30)
            result = vc.embed(texts=texts, model=schema.EMBEDDING_MODEL, input_type="document")
            vectors = result.embeddings

        for chunk, vector in zip(batch, vectors):
            doc = schema.build_embedding_doc(chunk=chunk, vector=vector)
            embeddings_collection.insert_one(doc)
            stats["inserted"] += 1

        print(f"  Embedded batch {batch_idx + 1}/{len(batches)} "
              f"({len(batch)} chunks, {stats['inserted']}/{len(to_embed)} total)")

        if batch_idx < len(batches) - 1:
            time.sleep(VOYAGE_BATCH_DELAY)

    # Orphan cleanup
    embeddings_collection.delete_many({
        "$or": [
            {"model_version": {"$ne": schema.EMBEDDING_MODEL_VERSION}},
            {"dimensions": {"$ne": schema.EMBEDDING_DIMENSIONS}},
        ]
    })

    print(f"  Embeddings: {stats['inserted']} inserted, {stats['replaced']} replaced, "
          f"{stats['skipped']} skipped")
    return stats

# ---------------------------------------------------------------------------
# Vector search index
# ---------------------------------------------------------------------------


def _ensure_vector_index(db, embeddings_coll: str) -> str:
    """Create or verify the vector search index. Returns status string."""
    collection = db[embeddings_coll]

    # Check for existing index
    existing_def = None
    existing_status = None
    try:
        indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
    except OperationFailure:
        indexes = []

    for idx in indexes:
        if idx.get("name") == schema.VECTOR_INDEX_NAME:
            existing_status = idx.get("status")
            existing_def = idx.get("latestDefinition") or idx.get("definition", {})
            break

    if existing_def is not None:
        existing_fields = existing_def.get("mappings", {}).get("fields", {})
        vector_field = existing_fields.get("vector", {})

        compatible = (
            vector_field.get("type") == "knnVector"
            and vector_field.get("dimensions") == 1024
            and vector_field.get("similarity") == "cosine"
        )
        if compatible:
            if existing_status == "READY":
                return "READY"
            return f"PENDING (status: {existing_status})"

        # Incompatible — report and stop
        diff_parts = []
        if vector_field.get("dimensions") != 1024:
            diff_parts.append(f"dimensions={vector_field.get('dimensions')} (expected 1024)")
        if vector_field.get("similarity") != "cosine":
            diff_parts.append(f"similarity={vector_field.get('similarity')} (expected cosine)")
        if vector_field.get("type") != "knnVector":
            diff_parts.append(f"type={vector_field.get('type')} (expected knnVector)")

        raise SystemExit(
            f"Vector index '{schema.VECTOR_INDEX_NAME}' exists but is incompatible: "
            f"{'; '.join(diff_parts)}. "
            "Drop it manually in the Atlas UI if you need to recreate."
        )

    # Create new index
    try:
        collection.database.command(
            "createSearchIndexes",
            embeddings_coll,
            indexes=[{
                "name": schema.VECTOR_INDEX_NAME,
                "type": "vectorSearch",
                "definition": schema.VECTOR_INDEX_DEFINITION,
            }],
        )
    except OperationFailure as exc:
        raise SystemExit(f"Failed to create vector search index: {exc}") from exc

    # Wait up to 60s for READY
    waited = 0
    while waited < 60:
        time.sleep(5)
        waited += 5
        try:
            indexes = list(collection.aggregate([{"$listSearchIndexes": {}}]))
            for idx in indexes:
                if idx.get("name") == schema.VECTOR_INDEX_NAME:
                    if idx.get("status") == "READY":
                        return "READY"
                    existing_status = idx.get("status")
        except OperationFailure:
            pass
        print(f"  Waiting for index... ({waited}s, status: {existing_status})")

    print(f"  Index '{schema.VECTOR_INDEX_NAME}' timed out waiting for READY. "
          f"Last status: {existing_status}. "
          "Check manually: db.embeddings.aggregate([{$listSearchIndexes: {}}])")
    return f"TIMEOUT (last status: {existing_status})"

# ---------------------------------------------------------------------------
# Sample query
# ---------------------------------------------------------------------------


def _sample_vector_query(db, settings) -> str:
    """Run a sample $vectorSearch query. Returns description string."""
    if not settings.voyage_api_key:
        return "skipped (no VOYAGE_API_KEY)"

    vc = VoyageClient(api_key=settings.voyage_api_key)
    try:
        result = vc.embed(
            texts=["punishment for theft"],
            model=schema.EMBEDDING_MODEL,
            input_type="query",
        )
        query_vector = result.embeddings[0]
    except PyMongoError as exc:
        return f"skipped (Voyage query embedding failed: {exc})"

    embeddings_coll = db[schema.EMBEDDINGS_COLLECTION]
    try:
        agg_result = list(embeddings_coll.aggregate([
            {
                "$vectorSearch": {
                    "index": schema.VECTOR_INDEX_NAME,
                    "path": "vector",
                    "queryVector": query_vector,
                    "numCandidates": 10,
                    "limit": 1,
                }
            },
            {
                "$project": {"chunk_id": 1, "_id": 0}
            },
        ]))
    except OperationFailure as exc:
        if "index" in str(exc).lower() and "not ready" in str(exc).lower():
            return "pending — index not ready"
        return f"error: {exc}"

    if not agg_result:
        return "no results"

    top_chunk_id = agg_result[0]["chunk_id"]
    chunk = db[schema.CHUNKS_COLLECTION].find_one({"_id": top_chunk_id})
    heading = (chunk or {}).get("section_id", "unknown")
    return f"top chunk: {top_chunk_id} (section: {heading})"

# ---------------------------------------------------------------------------
# Doc comparison
# ---------------------------------------------------------------------------

_EXCLUDE_KEYS = frozenset({"_record_index", "updated_at", "provenance", "created_at"})


def _docs_equal(existing: dict, new_doc: dict, extra_skip: tuple[str, ...] = ()) -> bool:
    """Return True if *existing* and *new_doc* are equivalent for upsert purposes."""
    skip = _EXCLUDE_KEYS | frozenset(extra_skip)
    for key, value in new_doc.items():
        if key in skip:
            continue
        if existing.get(key) != value:
            return False
    return True

# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(settings=None):
    """Execute the full ingestion pipeline."""
    if settings is None:
        settings = get_settings()

    print("=== MongoDB RAG Ingestion ===")
    print()

    # 1. Check MongoDB
    print("1. Checking MongoDB prerequisites...")
    client = _check_mongodb(settings)
    db_name = settings.mongodb_db_name if settings.app_env != "testing" else settings.mongodb_test_db_name
    db = client[db_name]
    print(f"   Connected to {db_name} on MongoDB {client.server_info().get('version', 'unknown')}")
    print()

    # 2. Read corpus
    print("2. Loading corpus...")
    bns_records = _load_jsonl(BNS_CORPUS)
    ipc_records = _load_jsonl(IPC_CORPUS)
    all_records = bns_records + ipc_records
    print(f"   BNS: {len(bns_records)} sections, IPC: {len(ipc_records)} sections")
    print()

    # 3. Ordinary indexes
    print("3. Ensuring ordinary indexes...")
    schema.ensure_ordinary_indexes(db)
    print("   Done.")
    print()

    # 4. Ingest sources
    print("4. Ingesting sources...")
    source_stats = _ingest_sources(db, schema.SOURCES_COLLECTION, bns_records, ipc_records)
    print(f"   {source_stats}")
    print()

    # 5. Ingest sections
    print("5. Ingesting sections...")
    section_stats = _ingest_sections(db, schema.SECTIONS_COLLECTION, all_records)
    print(f"   {section_stats}")
    print()

    # 6. Chunk sections
    print("6. Chunking sections...")
    chunk_count, all_chunk_ids = _chunk_sections(db, schema.SECTIONS_COLLECTION, schema.CHUNKS_COLLECTION)
    print()

    # 7. Embed chunks
    print("7. Embedding chunks...")
    embed_stats = _embed_chunks(db, schema.EMBEDDINGS_COLLECTION, all_chunk_ids, settings)
    print()

    # 8. Vector index
    print("8. Vector search index...")
    index_status = _ensure_vector_index(db, schema.EMBEDDINGS_COLLECTION)
    print(f"   Index status: {index_status}")
    print()

    # 9. Sample query
    print("9. Sample vector query...")
    query_result = _sample_vector_query(db, settings)
    print(f"   {query_result}")
    print()

    # 10. Verification
    print("10. Lightweight verification...")
    _print_verification(db, chunk_count, embed_stats)
    print()

    print("=== Ingestion complete ===")

    return {
        "source_stats": source_stats,
        "section_stats": section_stats,
        "chunk_count": chunk_count,
        "embed_stats": embed_stats,
        "index_status": index_status,
        "query_result": query_result,
    }


def _print_verification(db, chunk_count: int, embed_stats: dict):
    """Run classroom-sized verification checks."""
    # Collection counts
    sources_count = db[schema.SOURCES_COLLECTION].count_documents({})
    sections_count = db[schema.SECTIONS_COLLECTION].count_documents({})
    chunks_count = db[schema.CHUNKS_COLLECTION].count_documents({})
    embeddings_count = db[schema.EMBEDDINGS_COLLECTION].count_documents({})
    print(f"   Collections: sources={sources_count}, sections={sections_count}, "
          f"chunks={chunks_count}, embeddings={embeddings_count}")

    # Section-to-chunk link
    chunks_for_bns1 = list(db[schema.CHUNKS_COLLECTION].find({"section_id": "bns:1"}))
    print(f"   bns:1 chunks: {len(chunks_for_bns1)}")

    # Embedding dimension check
    sample_embedding = db[schema.EMBEDDINGS_COLLECTION].find_one()
    if sample_embedding:
        vec_len = len(sample_embedding.get("vector", []))
        print(f"   Sample embedding dimension: {vec_len} (expected 1024)")
    else:
        print("   No embeddings found")


if __name__ == "__main__":
    run()