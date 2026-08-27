import re
import sqlite3

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    DB_PATH,
    EMBED_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME,
)


model = SentenceTransformer(EMBED_MODEL)

client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))


def build_index():
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute(
        """
        SELECT identifier, search_text
        FROM series_catalogue
        """
    ).fetchall()

    conn.close()

    if not rows:
        print("No records found in SQLite.")
        return

    ids = [row[0] for row in rows]
    documents = [row[1] or "" for row in rows]

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(f"Index built: {len(rows)} records")


def search(query, k=10, filters=None):
    query = query.strip()

    if not query:
        return []

    # Get vector candidates
    query_embedding = model.encode(query).tolist()

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(50, max(k, 10)),
        include=["distances"]
    )

    vector_ids = vector_results["ids"][0]
    vector_distances = vector_results["distances"][0]

    vector_score = {
        identifier: 1 / (1 + distance)
        for identifier, distance
        in zip(vector_ids, vector_distances)
    }

    # Read records from SQLite
    conn = sqlite3.connect(DB_PATH)

    sql = """
        SELECT
            identifier,
            title,
            category,
            subcategory,
            frequency,
            unit,
            search_text
        FROM series_catalogue
        WHERE 1=1
    """

    params = []

    if filters:
        for field, value in filters.items():
            allowed = {
                "category",
                "subcategory",
                "frequency",
                "unit",
                "currency",
                "discontinued"
            }

            field = field.lower()

            if field not in allowed:
                continue

            sql += f" AND LOWER({field}) = LOWER(?)"
            params.append(str(value))

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    query_tokens = tokenize(query)

    ranked = []

    for row in rows:
        (
            identifier,
            title,
            category,
            subcategory,
            frequency,
            unit,
            search_text
        ) = row

        text_tokens = tokenize(search_text or "")

        overlap = len(query_tokens & text_tokens)

        lexical_score = (
            overlap / len(query_tokens)
            if query_tokens
            else 0
        )

        # Strong bonus when query words appear in title
        title_tokens = tokenize(title or "")
        title_overlap = len(query_tokens & title_tokens)

        title_score = (
            title_overlap / len(query_tokens)
            if query_tokens
            else 0
        )

        # Combine lexical and semantic relevance
        semantic_score = vector_score.get(identifier, 0)

        final_score = (
            0.50 * lexical_score +
            0.30 * title_score +
            0.20 * semantic_score
        )

        if final_score > 0:
            ranked.append(
                (
                    final_score,
                    identifier,
                    title,
                    category,
                    subcategory,
                    frequency,
                    unit
                )
            )

    # Deterministic ordering:
    # relevance first, identifier as tie-breaker
    ranked.sort(
        key=lambda x: (-x[0], x[1])
    )

    # If the top record is a high-confidence exact match (score >= 0.90),
    # prune lower-tier candidates that fall below score threshold
    if ranked and ranked[0][0] >= 0.90:
        top_score = ranked[0][0]
        ranked = [r for r in ranked if r[0] >= top_score * 0.92]

    return [
        {
            "identifier": row[1],
            "title": row[2],
            "category": row[3],
            "subcategory": row[4],
            "frequency": row[5],
            "unit": row[6],
            "score": round(row[0], 4)
        }
        for row in ranked[:k]
    ]


if __name__ == "__main__":
    build_index()

    while True:
        query = input("\nSearch: ").strip()

        if not query:
            break

        results = search(query)

        for result in results:
            print(result)