import duckdb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "sa_talent.duckdb"


def get_connection(path: str | Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection. Creates the file if it doesn't exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create tables if they don't already exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_postings (
            id          INTEGER PRIMARY KEY,
            title       TEXT,
            company     TEXT,
            location    TEXT,
            province    TEXT,
            industry    TEXT,
            description TEXT,
            source      TEXT,
            date_scraped TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_features (
            id              INTEGER PRIMARY KEY,
            title           TEXT,
            industry        TEXT,
            province        TEXT,
            skill_count     INTEGER,
            requires_degree INTEGER,
            skills          TEXT,
            date_scraped    TIMESTAMP
        )
    """)
