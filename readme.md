# Technical Exercise Submission: Series Catalogue Search & Retrieval Engine

An end-to-end system for ingesting, cleaning, indexing, and retrieving time series metadata using a hybrid approach that combines SQL-based filtering and semantic vector search.

---

## 1. Quickstart & Execution Guide

### Environment Setup

Activate your Python virtual environment:

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

```bat
# Command Prompt
.\venv\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

#### 1. Data Ingestion (SQLite)

```bash
python ingest.py --input data/series_catalogue_raw.csv
```

Expected output:

```text
Loaded XXX records into SQLite
```

#### 2. Build Index + Run Search CLI

```bash
python search.py
```

Expected output:

```text
Index built: XXX records
Search:
```

#### 3. Agent Interface

```bash
python agent.py
```

---

## 2. Project Structure

```text
Assignment/
├── agent.py
├── config.py
├── ingest.py
├── search.py
├── requirements.txt
├── README.md
├── catalogue.db
├── data/
│   └── series_catalogue_raw.csv
└── venv/
```

---

## 3. Technologies Used

| Tool | Purpose |
| --- | --- |
| sqlite3 | Unified storage engine for metadata, hierarchy, FTS5 lexical index, and single source of truth (catalogue.db). |
| sentence-transformers | Embedding generation (all-MiniLM-L6-v2) |
| numpy | Mathematical operations for calculating cosine similarity between vectors in-memory |
| pandas | Data ingestion and cleaning |
| openpyxl | Excel file handling |
| re | Tokenization |
| argparse | CLI argument handling |

---

## 4. Ingestion & Storage Design

### File Handling

The ingestion layer supports both CSV and Excel-formatted files using header-based detection:

```python
if header == b"PK\x03\x04":
    return pd.read_excel(file_path, engine="openpyxl")
```

### Data Cleaning

- Normalize column names to lowercase and underscores.
- Handle missing values:
  - currency → "NA"
  - parent → "ROOT"
- Convert `discontinued` to boolean.
- Preserve the original title for display.

### Search Text Construction

A denormalized field is created for search:

```python
search_text = title + category + subcategory + subset + frequency + unit + currency
```

### SQLite Schema

```sql
CREATE TABLE series_catalogue (
    identifier TEXT PRIMARY KEY,
    parent TEXT,
    childtree1 TEXT,
    childtree1_name TEXT,
    childtree2 TEXT,
    childtree2_name TEXT,
    title TEXT,
    category TEXT,
    subcategory TEXT,
    subset TEXT,
    frequency TEXT,
    unit TEXT,
    currency TEXT,
    discontinued BOOLEAN,
    search_text TEXT
);
```

---

## 5. Search Architecture

The system uses a hybrid retrieval approach combining lexical and semantic signals.

### 1. Semantic Search

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Embeddings are computed during ingestion and stored directly as binary blobs in catalogue.db.
- The query vector is generated at search time and matched against stored vectors using cosine similarity.

### 2. Lexical Matching

Token overlap between the query and `search_text`:

```python
overlap = len(query_tokens & text_tokens)
lexical_score = overlap / len(query_tokens)
```

### 3. Title Relevance Boost

An additional score weight is applied when query tokens appear in the title.

### 4. Final Scoring

```python
final_score = (
    0.50 * lexical_score +
    0.30 * title_score +
    0.20 * semantic_score
)
```

### 5. Result Ranking

- Sorted by score in descending order.
- Tie-breaker: identifier in ascending order.

---

## 6. Indexing Strategy

The system uses SQLite persistent database storage (catalogue.db):

```python
# Save metadata and dense embeddings directly into SQLite table
cursor.execute(
    "INSERT OR REPLACE INTO series_catalogue VALUES (?, ?, ..., ?)",
    (*record_values, embedding_blob)
)
```
This ensures that all metadata, search texts, and vector embeddings reside in a single database file, supporting incremental updates without needing an external vector database service.

---

## 7. Agent Layer

Provides a structured interface over the search function:

```python
def agent(query):
    results = search(query)
    return {"query": query, "results": results}
```

- Returns only catalogue-backed results.
- Avoids hallucination and external data generation.

---

## 8. System Behavior

### Strengths

- Handles both exact keyword and semantic queries.
- Supports structured filtering such as category, frequency, and unit.
- Robust ingestion with file format detection.
- Lightweight and efficient for small datasets.

### Example Queries

- "IndiGo on-time performance at Delhi"
  - High lexical and title overlap.
  - Correct record appears at the top.

- "monthly cashew exports in dollars"
  - Prioritizes monthly over cumulative.
  - Prefers USD over rupees.

- "IndiGo punctuality"
  - Semantic matching maps "punctuality" to "on time performance".

- "how were flights in December?"
  - Returns metadata-level matches.
  - Does not provide time-series values, because they are not present in the dataset.

---

## 9. Design Decisions

### Why Hybrid Search?

| Approach | Limitation |
| --- | --- |
| Keyword-only | Fails on synonyms |
| Vector-only | Misses exact domain terms |
| Hybrid | Combines both strengths |

### Why Single-File SQLite Storage?

- **Zero Synchronization Lag:** Relational metadata, full-text indexes, and vector embeddings remain in one single file (`catalogue.db`).
- **Simplified Deployment:** No need to install, configure, or run a separate vector store like ChromaDB.
- **Data Integrity:** ACID compliance ensures transactional updates across metadata and embeddings.

---

## 10. Limitations & Future Improvements

### Current Limitations

- No typo correction (for example, "Indgo").
- Full table scan in SQLite during vector similarity scoring.
- No threshold filtering for low-confidence results.

### Future Enhancements

- Add fuzzy matching and spell correction.
- Use SQLite vector extensions (such as `sqlite-vec`) for accelerated vector indexing.
- Introduce score thresholding to ignore irrelevant queries.

---

## 11. Key Learnings

Building this system required handling real-world data challenges:

- File format inconsistencies (CSV vs Excel)
- Encoding issues
- Malformed rows during ingestion

The system was developed iteratively:

ingestion → cleaning → indexing → hybrid search
