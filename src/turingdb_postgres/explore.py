"""Interactive and one-shot exploration of the restored Postgres database."""

from __future__ import annotations

import psycopg
from psycopg import sql
from rich.console import Console
from rich.table import Table

from . import docker_pg
from .config import Settings

console = Console()

# How many characters of a single cell we print before truncating.
CELL_WIDTH = 100

HELP = """[bold]Commands[/]
  [cyan]tables[/]             list tables (schema, rows, size)
  [cyan]schemas[/]            list schemas
  [cyan]describe <table>[/]   describe a table's columns + constraints
  [cyan]sample <table>[/]     show sample rows from a table (uses current limit)
  [cyan]fk <c>.<col> -> <p>.<col>[/]  add a foreign key, e.g. fk orders.customer_id -> customers.id
  [cyan]limit <n>[/]          set the row limit for samples/queries (current: {limit})
  [cyan]help[/]               show this help
  [cyan]quit[/]               quit
  [dim]anything else is executed as SQL[/]
"""


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.conninfo(), autocommit=True)


def _truncate(value: object) -> str:
    text = "∅" if value is None else str(value)
    text = text.replace("\n", " ⏎ ")
    if len(text) > CELL_WIDTH:
        text = text[:CELL_WIDTH] + "…"
    return text


def render_rows(columns: list[str], rows: list[tuple], title: str | None = None) -> None:
    table = Table(title=title, show_lines=False, header_style="bold cyan", title_style="bold")
    for col in columns:
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(*(_truncate(v) for v in row))
    console.print(table)
    console.print(f"[dim]{len(rows)} row(s)[/]")


def list_schemas(conn: psycopg.Connection) -> None:
    rows = conn.execute(
        """
        SELECT nspname AS schema,
               pg_catalog.pg_get_userbyid(nspowner) AS owner
        FROM pg_namespace
        WHERE nspname NOT IN ('pg_catalog', 'information_schema')
          AND nspname NOT LIKE 'pg_%'
        ORDER BY 1
        """
    ).fetchall()
    render_rows(["schema", "owner"], rows, title="Schemas")


def list_tables(conn: psycopg.Connection) -> None:
    rows = conn.execute(
        """
        SELECT n.nspname AS schema,
               c.relname AS table,
               c.reltuples::bigint AS est_rows,
               pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE c.relkind = 'r'
          AND n.nspname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY pg_total_relation_size(c.oid) DESC
        """
    ).fetchall()
    render_rows(["schema", "table", "est. rows", "total size"], rows, title="Tables")


def _resolve_table(conn: psycopg.Connection, name: str) -> tuple[str, str] | None:
    """Resolve a possibly schema-qualified table name to (schema, table)."""
    if "." in name:
        schema, table = name.split(".", 1)
        row = conn.execute(
            """SELECT 1 FROM information_schema.tables
               WHERE table_schema = %s AND table_name = %s""",
            (schema, table),
        ).fetchone()
        return (schema, table) if row else None

    matches = conn.execute(
        """SELECT table_schema, table_name FROM information_schema.tables
           WHERE table_name = %s
             AND table_schema NOT IN ('pg_catalog', 'information_schema')""",
        (name,),
    ).fetchall()
    if not matches:
        return None
    if len(matches) > 1:
        console.print(
            f"[yellow]'{name}' is ambiguous:[/] "
            + ", ".join(f"{s}.{t}" for s, t in matches)
            + " — qualify it with the schema."
        )
        return None
    return matches[0][0], matches[0][1]


def describe_table(conn: psycopg.Connection, name: str) -> None:
    resolved = _resolve_table(conn, name)
    if not resolved:
        console.print(f"[red]No such table:[/] {name}")
        return
    schema, table = resolved

    cols = conn.execute(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    ).fetchall()
    render_rows(
        ["column", "type", "nullable", "default"],
        cols,
        title=f"{schema}.{table} — columns",
    )

    constraints = conn.execute(
        """
        SELECT conname AS name,
               CASE contype WHEN 'p' THEN 'PRIMARY KEY'
                            WHEN 'f' THEN 'FOREIGN KEY'
                            WHEN 'u' THEN 'UNIQUE'
                            WHEN 'c' THEN 'CHECK'
                            ELSE contype::text END AS type,
               pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid = %s::regclass
        ORDER BY contype
        """,
        (f"{schema}.{table}",),
    ).fetchall()
    if constraints:
        render_rows(["name", "type", "definition"], constraints, title="Constraints")


def sample_table(conn: psycopg.Connection, name: str, limit: int) -> None:
    resolved = _resolve_table(conn, name)
    if not resolved:
        console.print(f"[red]No such table:[/] {name}")
        return
    schema, table = resolved
    cur = conn.execute(
        sql.SQL("SELECT * FROM {}.{} LIMIT {}").format(
            sql.Identifier(schema), sql.Identifier(table), sql.Literal(limit)
        )
    )
    columns = [d.name for d in cur.description]
    render_rows(columns, cur.fetchall(), title=f"{schema}.{table} — first {limit} rows")


def run_sql(conn: psycopg.Connection, query: str, limit: int) -> None:
    try:
        cur = conn.execute(query.encode())
    except psycopg.Error as exc:
        console.print(f"[red]SQL error:[/] {exc}")
        return
    if cur.description is None:
        # non-SELECT statement (INSERT/UPDATE/DDL/...)
        console.print(f"[green]OK[/] — {cur.statusmessage}")
        return
    columns = [d.name for d in cur.description]
    rows = cur.fetchmany(limit)
    render_rows(columns, rows)
    if len(rows) == limit:
        console.print(f"[dim](output capped at limit={limit}; use \\limit to change)[/]")


def add_fk(settings: Settings, spec: str) -> None:
    """Add a foreign key from an arrow-syntax spec, e.g. 'a.col -> b.col'."""
    try:
        parsed = docker_pg.parse_fk_spec(spec)
    except ValueError as exc:
        console.print(f"[yellow]usage:[/] fk <child.col> -> <parent.col>  ([dim]{exc}[/])")
        return
    try:
        docker_pg.apply_foreign_keys(settings, [parsed])
    except docker_pg.DockerError as exc:
        console.print(f"[red]✗ {exc}[/]")


def interactive(settings: Settings, limit: int) -> None:
    """A small psql-like REPL over the restored database."""
    conn = connect(settings)
    console.print(
        f"[bold green]Postgres explorer[/] — db [bold]{settings.dbname}[/] "
        f"@ {settings.host}:{settings.port}"
    )
    console.print("Type [cyan]help[/] for commands, [cyan]quit[/] to exit.\n")
    list_tables(conn)

    prompt = f"[bold]{settings.dbname}>[/] "
    while True:
        try:
            raw = console.input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye")
            break
        if not raw:
            continue

        # A command is the first word (case-insensitive); the rest is its argument.
        # Anything whose first word isn't a known command is executed as SQL.
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            console.print("bye")
            break
        elif cmd in ("help", "?"):
            console.print(HELP.format(limit=limit))
        elif cmd == "tables":
            list_tables(conn)
        elif cmd == "schemas":
            list_schemas(conn)
        elif cmd in ("describe", "desc"):
            if arg:
                describe_table(conn, arg)
            else:
                console.print("[yellow]usage:[/] describe <table>")
        elif cmd == "sample":
            if arg:
                sample_table(conn, arg, limit)
            else:
                console.print("[yellow]usage:[/] sample <table>")
        elif cmd == "fk":
            if arg:
                add_fk(settings, arg)
            else:
                console.print("[yellow]usage:[/] fk <child.col> -> <parent.col>")
        elif cmd == "limit":
            if arg.isdigit():
                limit = int(arg)
                console.print(f"[green]✓[/] limit set to {limit}")
            else:
                console.print(f"current limit is {limit}  ([dim]usage: limit <n>[/])")
        else:
            run_sql(conn, raw, limit)

    conn.close()
