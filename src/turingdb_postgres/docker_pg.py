"""Manage a dockerised PostgreSQL instance and restore a dump into it.

We use Docker because the host has no native `psql`/`pg_restore`. The dump's
directory is bind-mounted read-only into the container so a 1 GB+ file never
needs copying, and `pg_restore` runs *inside* the container against the mounted
path.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import psycopg
from rich.console import Console

from .config import Settings

console = Console()

# Path at which the dump's parent directory is mounted inside the container.
MOUNT_POINT = "/dump"


class DockerError(RuntimeError):
    pass


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def ensure_docker() -> None:
    """Fail early with a clear message if Docker is unusable."""
    proc = _run(["docker", "info"])
    if proc.returncode != 0:
        raise DockerError(
            "Docker is not available or the daemon is not running.\n" + proc.stderr.strip()
        )


def detect_dbname(image: str, dump: Path) -> str | None:
    """Read the source database name from a custom-format dump's archive header.

    `pg_restore -l` prints header comment lines like ";     dbname: mydb".
    Returns None if the name can't be read (e.g. a plain-SQL dump, on which
    `pg_restore -l` fails).
    """
    proc = _run(
        [
            "docker", "run", "--rm",
            "-v", f"{dump.parent}:{MOUNT_POINT}:ro",
            image,
            "pg_restore", "-l", f"{MOUNT_POINT}/{dump.name}",
        ]
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip().lstrip(";").strip()
        if stripped.lower().startswith("dbname:"):
            name = stripped.split(":", 1)[1].strip()
            return name or None
    return None


def resolve_dbname(image: str, dump: Path, override: str | None) -> tuple[str, str]:
    """Decide the target database name and report where it came from.

    Precedence: explicit --dbname flag → the dump's embedded name → the dump
    filename (sanitised to a valid identifier).
    """
    if override:
        return override, "--dbname flag"
    detected = detect_dbname(image, dump)
    if detected:
        return detected, "dump header"
    stem = re.sub(r"[^A-Za-z0-9_]", "_", dump.stem).strip("_").lower()
    return (stem or "imported"), "dump filename"


def container_state(name: str) -> str:
    """Return 'running', 'stopped', or 'absent' for the named container."""
    proc = _run(["docker", "inspect", "-f", "{{.State.Running}}", name])
    if proc.returncode != 0:
        return "absent"
    return "running" if proc.stdout.strip() == "true" else "stopped"


def start_container(settings: Settings, dump_dir: Path) -> None:
    """Create-and-start (or start) the Postgres container.

    The container is created with the dump's directory bind-mounted read-only.
    """
    state = container_state(settings.container)
    if state == "running":
        console.print(f"[green]✓[/] container [bold]{settings.container}[/] already running")
        return
    if state == "stopped":
        console.print(f"Starting existing container [bold]{settings.container}[/]…")
        proc = _run(["docker", "start", settings.container])
        if proc.returncode != 0:
            raise DockerError(f"failed to start container: {proc.stderr.strip()}")
        return

    console.print(
        f"Creating container [bold]{settings.container}[/] "
        f"([cyan]{settings.image}[/]) on port [cyan]{settings.port}[/]…"
    )
    proc = _run(
        [
            "docker", "run", "-d",
            "--name", settings.container,
            "-e", f"POSTGRES_USER={settings.user}",
            "-e", f"POSTGRES_PASSWORD={settings.password}",
            "-e", "POSTGRES_DB=postgres",
            "-p", f"{settings.port}:5432",
            "-v", f"{dump_dir}:{MOUNT_POINT}:ro",
            settings.image,
        ]
    )
    if proc.returncode != 0:
        raise DockerError(f"failed to create container: {proc.stderr.strip()}")


def stop_container(settings: Settings, remove: bool = False) -> None:
    """Stop (and optionally remove) the managed container."""
    _run(["docker", "stop", settings.container])
    if remove:
        _run(["docker", "rm", settings.container])


def wait_ready(settings: Settings, timeout: float = 60.0) -> None:
    """Block until Postgres accepts connections (or timeout)."""
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    with console.status("Waiting for Postgres to accept connections…"):
        while time.monotonic() < deadline:
            try:
                with psycopg.connect(settings.conninfo("postgres"), connect_timeout=3):
                    return
            except Exception as exc:  # noqa: BLE001 — any connection error means "not ready yet"
                last_err = exc
                time.sleep(1.0)
    raise DockerError(f"Postgres did not become ready within {timeout:.0f}s: {last_err}")


def database_exists(settings: Settings, dbname: str) -> bool:
    with psycopg.connect(settings.conninfo("postgres"), autocommit=True) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)
        ).fetchone()
        return row is not None


def create_database(settings: Settings, recreate: bool) -> None:
    """Ensure the target database exists, optionally dropping it first."""
    from psycopg import sql

    with psycopg.connect(settings.conninfo("postgres"), autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.dbname,)
        ).fetchone()
        if exists and recreate:
            console.print(f"Dropping existing database [bold]{settings.dbname}[/]…")
            # terminate other backends so DROP doesn't block
            conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (settings.dbname,),
            )
            conn.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(settings.dbname)))
            exists = False
        if not exists:
            console.print(f"Creating database [bold]{settings.dbname}[/]…")
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.dbname)))
        else:
            console.print(
                f"Database [bold]{settings.dbname}[/] already exists "
                "(use --recreate to start fresh)."
            )


def restore(settings: Settings, dump_filename: str, jobs: int = 4) -> None:
    """Run pg_restore inside the container against the bind-mounted dump."""
    container_path = f"{MOUNT_POINT}/{dump_filename}"
    cmd = [
        "docker", "exec", settings.container,
        "pg_restore",
        "-U", settings.user,
        "-d", settings.dbname,
        "--no-owner",         # the dump may be owned by a role that doesn't exist locally
        "--no-privileges",    # skip GRANT/REVOKE that reference such roles
        "--no-comments",
        "--exit-on-error",
        "-j", str(jobs),
        "-v",
        container_path,
    ]
    console.print(f"Restoring [cyan]{dump_filename}[/] with [cyan]{jobs}[/] parallel jobs…")
    console.print("[dim]$ " + " ".join(cmd) + "[/]")

    # Stream pg_restore's progress (verbose mode prints one line per object).
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("pg_restore: processing data for table"):
            console.print(f"  [green]→[/] {line.split('table', 1)[1].strip()}")
        elif "error" in line.lower() or "warning" in line.lower():
            console.print(f"  [yellow]{line}[/]")
    rc = proc.wait()
    if rc != 0:
        raise DockerError(f"pg_restore exited with code {rc}")
    console.print("[green]✓[/] restore complete")


# --------------------------------------------------------------------------
# Foreign keys
#
# A dump may ship only primary keys, but a column like `<x>_id` often clearly
# references another table. FK constraints are what the graph importer reads to
# derive edges, so `load` (re)establishes them here — either from an explicit
# spec or by inference validated against the data.
# --------------------------------------------------------------------------

# A parsed FK: (child_table, child_col, parent_table, parent_col).
FKSpec = tuple[str, str, str, str]


def parse_fk_spec(spec: str) -> FKSpec:
    """Parse 'child.col=parent.col' (also '->' or ':') into an FKSpec.

    Table names may be schema-qualified (e.g. 'public.orders').
    """
    sep = next((s for s in ("->", "=", ":") if s in spec), None)
    if sep is None:
        raise ValueError(f"FK must look like 'child.col=parent.col', got '{spec}'")
    left, right = (part.strip() for part in spec.split(sep, 1))

    def split_col(side: str) -> tuple[str, str]:
        table, _, col = side.rpartition(".")
        if not table or not col:
            raise ValueError(f"expected TABLE.COLUMN, got '{side}'")
        return table, col

    ct, cc = split_col(left)
    pt, pc = split_col(right)
    return ct, cc, pt, pc


def _qualified(name: str):
    """A sql.Identifier for a possibly schema-qualified name."""
    from psycopg import sql

    return sql.SQL(".").join(sql.Identifier(p) for p in name.split("."))


def _resolve_table(conn: psycopg.Connection, name: str) -> str:
    """Return a 'schema.table' string; look the schema up if unqualified."""
    if "." in name:
        return name
    matches = conn.execute(
        """SELECT table_schema FROM information_schema.tables
           WHERE table_name = %s
             AND table_schema NOT IN ('pg_catalog', 'information_schema')""",
        (name,),
    ).fetchall()
    if len(matches) == 1:
        return f"{matches[0][0]}.{name}"
    if not matches:
        raise DockerError(f"no such table: {name}")
    raise DockerError(f"table '{name}' is ambiguous across schemas — qualify it")


def _fk_already_present(conn, child: str, child_col: str, parent: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM pg_constraint c
        WHERE c.contype = 'f'
          AND c.conrelid = %s::regclass
          AND c.confrelid = %s::regclass
          AND %s = ANY (
              SELECT a.attname FROM pg_attribute a
              WHERE a.attrelid = c.conrelid AND a.attnum = ANY (c.conkey))
        """,
        (child, parent, child_col),
    ).fetchone()
    return row is not None


def apply_foreign_keys(settings: Settings, specs: list[FKSpec]) -> int:
    """Add each FK (idempotently). Returns how many were newly created."""
    from psycopg import sql

    added = 0
    with psycopg.connect(settings.conninfo(), autocommit=True) as conn:
        for ct, cc, pt, pc in specs:
            child = _resolve_table(conn, ct)
            parent = _resolve_table(conn, pt)
            arrow = f"{child}.{cc} → {parent}.{pc}"
            if _fk_already_present(conn, child, cc, parent):
                console.print(f"[dim]• FK already present: {arrow}[/]")
                continue
            name = f"{child.split('.')[-1]}_{cc}_fkey"
            stmt = sql.SQL(
                "ALTER TABLE {child} ADD CONSTRAINT {name} "
                "FOREIGN KEY ({ccol}) REFERENCES {parent} ({pcol})"
            ).format(
                child=_qualified(child),
                name=sql.Identifier(name),
                ccol=sql.Identifier(cc),
                parent=_qualified(parent),
                pcol=sql.Identifier(pc),
            )
            try:
                conn.execute(stmt)
            except psycopg.Error as exc:
                raise DockerError(f"could not add FK {arrow}: {exc}") from exc
            console.print(f"[green]✓[/] added FK {arrow}")
            added += 1
    return added


def _candidate_parents(prefix: str) -> list[str]:
    """Plausible parent table names for a column '<prefix>_id'."""
    cands = [prefix, prefix + "s", prefix + "es"]
    if prefix.endswith("y"):
        cands.append(prefix[:-1] + "ies")
    return cands


def infer_foreign_keys(settings: Settings) -> list[FKSpec]:
    """Discover missing FKs by convention, validated against the data.

    For every `<x>_id` column we look for a table named like `<x>` (or a simple
    plural) whose single-column primary/unique key the column's values all
    satisfy. Only relationships that hold for every non-null row are returned,
    so we never propose a FK the data would reject.
    """
    from psycopg import sql

    inferred: list[FKSpec] = []
    with psycopg.connect(settings.conninfo()) as conn:
        # Single-column primary/unique keys → referenceable (schema, table, keycol).
        keys = conn.execute(
            """
            SELECT n.nspname, c.relname, a.attname
            FROM pg_constraint con
            JOIN pg_class c ON c.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = con.conkey[1]
            WHERE con.contype IN ('p', 'u')
              AND array_length(con.conkey, 1) = 1
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            """
        ).fetchall()
        # index by lower(table name) → list of (schema, table, keycol)
        refs: dict[str, list[tuple[str, str, str]]] = {}
        for schema, table, keycol in keys:
            refs.setdefault(table.lower(), []).append((schema, table, keycol))

        cols = conn.execute(
            """
            SELECT table_schema, table_name, column_name
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND column_name LIKE '%\\_id'
            """
        ).fetchall()

        for schema, table, col in cols:
            prefix = col[:-3]  # strip '_id'
            candidates: list[tuple[str, str, str]] = []
            for cand in _candidate_parents(prefix):
                candidates.extend(refs.get(cand.lower(), []))
            # de-dupe, prefer a parent in the same schema
            candidates.sort(key=lambda r: (r[0] != schema,))

            for p_schema, p_table, p_key in candidates:
                if (p_schema, p_table) == (schema, table):
                    continue  # skip self-reference
                child_fq = f"{schema}.{table}"
                parent_fq = f"{p_schema}.{p_table}"
                if _fk_already_present(conn, child_fq, col, parent_fq):
                    break
                n_nonnull, n_orphans = conn.execute(
                    sql.SQL(
                        "SELECT "
                        "(SELECT count(*) FROM {child} WHERE {ccol} IS NOT NULL), "
                        "(SELECT count(*) FROM {child} ch WHERE ch.{ccol} IS NOT NULL "
                        " AND NOT EXISTS (SELECT 1 FROM {parent} p WHERE p.{pcol} = ch.{ccol}))"
                    ).format(
                        child=_qualified(child_fq),
                        ccol=sql.Identifier(col),
                        parent=_qualified(parent_fq),
                        pcol=sql.Identifier(p_key),
                    )
                ).fetchone()
                if n_nonnull > 0 and n_orphans == 0:
                    inferred.append((child_fq, col, parent_fq, p_key))
                    break  # first validated parent wins
    return inferred


def table_summary(settings: Settings) -> list[tuple[str, str, int]]:
    """Return (schema, table, exact_row_count) for every user table."""
    from psycopg import sql

    out: list[tuple[str, str, int]] = []
    with psycopg.connect(settings.conninfo()) as conn:
        tables = conn.execute(
            """
            SELECT n.nspname, c.relname
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY 1, 2
            """
        ).fetchall()
        for schema, table in tables:
            count = conn.execute(
                sql.SQL("SELECT count(*) FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            ).fetchone()[0]
            out.append((schema, table, count))
    return out
