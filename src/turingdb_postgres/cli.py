"""Command-line interface for importing a Postgres dump into TuringDB.

Subcommands:
  load    restore a Postgres dump into a dockerised Postgres instance
  show    explore the restored database interactively (or one-shot)
  fk      add foreign key constraint(s) to the loaded database
  graph   build a TuringDB graph (tables -> nodes, foreign keys -> edges)
  vector  build a TuringDB vector index from embeddings stored in Postgres
  status  show the state of the managed Postgres container
  stop    stop / remove the managed container
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import docker_pg, explore
from .api import PostgresDump
from .config import (
    DEFAULT_CONTAINER,
    DEFAULT_IMAGE,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
    Settings,
)

console = Console()

app = typer.Typer(
    help="Import a PostgreSQL dump into a TuringDB graph.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def load(
    dump: Path = typer.Argument(..., help="Path to the Postgres dump file (custom format)."),
    dbname: Optional[str] = typer.Option(
        None,
        help="Target database name. Default: auto-detected from the dump header, "
        "falling back to the dump filename.",
    ),
    container: str = typer.Option(DEFAULT_CONTAINER, help="Docker container name."),
    image: str = typer.Option(DEFAULT_IMAGE, help="Postgres Docker image."),
    port: int = typer.Option(DEFAULT_PORT, help="Host port mapped to Postgres 5432."),
    user: str = typer.Option(DEFAULT_USER, help="Postgres superuser."),
    password: str = typer.Option(DEFAULT_PASSWORD, help="Postgres superuser password."),
    jobs: int = typer.Option(4, "--jobs", "-j", help="Parallel pg_restore jobs."),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop the target database first if it exists."
    ),
    fk: Optional[list[str]] = typer.Option(
        None,
        "--fk",
        help="Add a foreign key after restore, e.g. "
        "'orders.customer_id=customers.id'. Repeatable.",
    ),
    infer_fks: bool = typer.Option(
        True,
        "--infer-fks/--no-infer-fks",
        help="Auto-detect and add missing foreign keys by naming convention "
        "(each validated against the data before it is created).",
    ),
) -> None:
    """Spin up a dockerised Postgres and restore DUMP into it."""
    dump = dump.expanduser().resolve()
    if not dump.is_file():
        console.print(f"[red]Dump file not found:[/] {dump}")
        raise typer.Exit(1)

    try:
        docker_pg.ensure_docker()

        resolved_dbname, source = docker_pg.resolve_dbname(image, dump, dbname)
        console.print(f"Target database: [bold]{resolved_dbname}[/] [dim](from {source})[/]")
        settings = Settings(
            container=container,
            image=image,
            port=port,
            user=user,
            password=password,
            dbname=resolved_dbname,
        )

        docker_pg.start_container(settings, dump.parent)
        docker_pg.wait_ready(settings)
        docker_pg.create_database(settings, recreate=recreate)
        docker_pg.restore(settings, dump.name, jobs=jobs)

        # Establish foreign keys: explicit specs first, then inferred ones.
        try:
            explicit = [docker_pg.parse_fk_spec(s) for s in (fk or [])]
        except ValueError as exc:
            console.print(f"[red]✗ bad --fk value:[/] {exc}")
            raise typer.Exit(1)
        if explicit or infer_fks:
            console.print("\n[bold]Foreign keys[/]")
            docker_pg.apply_foreign_keys(settings, explicit)
            if infer_fks:
                docker_pg.apply_foreign_keys(settings, docker_pg.infer_foreign_keys(settings))
    except docker_pg.DockerError as exc:
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(1)

    settings.save()

    console.print()
    summary = docker_pg.table_summary(settings)
    from rich.table import Table

    tbl = Table(title="Restored tables", header_style="bold cyan")
    tbl.add_column("schema")
    tbl.add_column("table")
    tbl.add_column("rows", justify="right")
    for schema, table, count in summary:
        tbl.add_row(schema, table, f"{count:,}")
    console.print(tbl)
    console.print(
        f"\n[green]✓ Done.[/] Explore it with: "
        f"[bold cyan]tpg show[/]  (db '{settings.dbname}' @ {settings.host}:{settings.port})"
    )


@app.command()
def show(
    table: Optional[str] = typer.Option(None, "--table", "-t", help="Describe + sample a table, then exit."),
    sql: Optional[str] = typer.Option(None, "--sql", help="Run a single SQL query, then exit."),
    tables: bool = typer.Option(False, "--tables", help="List all tables, then exit."),
    limit: int = typer.Option(20, "--limit", "-n", help="Row limit for samples/queries."),
) -> None:
    """Explore the restored database (interactive REPL by default)."""
    settings = Settings.load()

    if docker_pg.container_state(settings.container) != "running":
        console.print(
            f"[yellow]Container '{settings.container}' is not running.[/] "
            "Run [bold cyan]tpg load <dump>[/] first."
        )
        raise typer.Exit(1)

    try:
        conn = explore.connect(settings)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not connect:[/] {exc}")
        raise typer.Exit(1)

    # One-shot modes.
    if tables:
        explore.list_tables(conn)
    elif table:
        explore.describe_table(conn, table)
        explore.sample_table(conn, table, limit)
    elif sql:
        explore.run_sql(conn, sql, limit)
    else:
        conn.close()
        explore.interactive(settings, limit)
        return
    conn.close()


@app.command()
def fk(
    spec: list[str] = typer.Argument(
        ...,
        help="Foreign key as 'child.col -> parent.col' (also = or :). Repeatable.",
    ),
) -> None:
    """Add foreign key constraint(s) to the loaded database."""
    settings = Settings.load()
    if docker_pg.container_state(settings.container) != "running":
        console.print(
            f"[yellow]Container '{settings.container}' is not running.[/] "
            "Run [bold cyan]tpg load <dump>[/] first."
        )
        raise typer.Exit(1)

    db = PostgresDump.from_state()
    added = 0
    for s in spec:
        try:
            added += db.add_foreign_key(s)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]✗ {exc}[/]")
            raise typer.Exit(1)
    console.print(f"[green]✓[/] {added} foreign key(s) added.")


@app.command()
def graph(
    name: str = typer.Argument(..., help="Name of the TuringDB graph to build."),
    host: str = typer.Option("http://localhost:6666", help="TuringDB server URL."),
    schema: Optional[list[str]] = typer.Option(
        None, "--schema", help="Only include tables in this schema. Repeatable."
    ),
    include: Optional[list[str]] = typer.Option(
        None, "--include", help="Only include this table (name or schema.table). Repeatable."
    ),
    exclude: Optional[list[str]] = typer.Option(
        None, "--exclude", help="Skip this table (name or schema.table). Repeatable."
    ),
    append: bool = typer.Option(
        False, "--append", help="Add to the existing graph instead of clearing it first."
    ),
    include_arrays: bool = typer.Option(
        False, "--include-arrays", help="Keep array-typed columns as node properties (skipped by default)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the derived node/edge mapping and exit without importing."
    ),
) -> None:
    """Build a TuringDB graph: one node label per table, one edge per foreign key."""
    from rich.table import Table

    from .graph import derive_specs

    settings = Settings.load()
    if docker_pg.container_state(settings.container) != "running":
        console.print(
            f"[yellow]Container '{settings.container}' is not running.[/] "
            "Run [bold cyan]tpg load <dump>[/] first."
        )
        raise typer.Exit(1)

    db = PostgresDump.from_state()
    try:
        nodes, edges, notes = derive_specs(
            db, schemas=schema, include=include, exclude=exclude, include_arrays=include_arrays
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not read the schema:[/] {exc}")
        raise typer.Exit(1)

    ntbl = Table(title="Nodes (from tables)", header_style="bold cyan")
    ntbl.add_column("label")
    ntbl.add_column("key")
    ntbl.add_column("properties", justify="right")
    for n in nodes:
        ntbl.add_row(n.label, n.key or "[dim]—[/]", str(len(n.properties or [])))
    console.print(ntbl)

    if edges:
        etbl = Table(title="Edges (from foreign keys)", header_style="bold cyan")
        etbl.add_column("relationship")
        for e in edges:
            etbl.add_row(f"({e.from_label})-[:{e.type}]->({e.to_label})")
        console.print(etbl)

    for note in notes:
        console.print(f"[yellow]• {note}[/]")

    if not nodes:
        console.print("[yellow]No tables selected — nothing to import.[/]")
        raise typer.Exit(1)

    if dry_run:
        console.print("\n[dim]dry run — nothing was imported.[/]")
        return

    console.print(f"\nImporting into TuringDB graph [bold]{name}[/] @ {host}…")
    try:
        stats = db.import_graph(name, nodes, edges, host=host, recreate=not append)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ import failed:[/] {exc}")
        raise typer.Exit(1)
    total_n = sum(stats["nodes"].values())
    total_e = sum(stats["edges"].values())
    console.print(
        f"[green]✓ Done.[/] {total_n:,} nodes, {total_e:,} edges in graph '{name}'."
    )


@app.command()
def vector(
    index: str = typer.Argument(..., help="Name of the TuringDB vector index to build."),
    sql: Optional[str] = typer.Option(
        None, "--sql", help="SQL returning (id, embedding); id must be an integer node key."
    ),
    table: Optional[str] = typer.Option(None, "--table", help="Table holding the vectors (with --key/--column)."),
    key: Optional[str] = typer.Option(None, "--key", help="Integer id column (joins results back to nodes)."),
    column: Optional[str] = typer.Option(None, "--column", help="Column holding the embedding array."),
    host: str = typer.Option("http://localhost:6666", help="TuringDB server URL."),
    graph: Optional[str] = typer.Option(None, "--graph", help="Set this graph as context before indexing."),
    metric: str = typer.Option("COSINE", help="Distance metric: COSINE or EUCLID."),
    dimension: Optional[int] = typer.Option(
        None, help="Vector dimension (auto-detected from the first row if omitted)."
    ),
    turing_dir: Optional[Path] = typer.Option(
        None, "--turing-dir", help="The TuringDB server's turing dir (default ~/.turing); "
        "the CSV is written to its data/ subdirectory.",
    ),
    limit: Optional[int] = typer.Option(None, help="Only index the first N vectors (for testing)."),
) -> None:
    """Build a TuringDB vector index from embedding vectors stored in Postgres."""
    settings = Settings.load()
    if docker_pg.container_state(settings.container) != "running":
        console.print(
            f"[yellow]Container '{settings.container}' is not running.[/] "
            "Run [bold cyan]tpg load <dump>[/] first."
        )
        raise typer.Exit(1)

    db = PostgresDump.from_state()
    try:
        result = db.build_vector_index(
            index,
            sql=sql,
            table=table,
            key=key,
            column=column,
            host=host,
            graph=graph,
            metric=metric,
            dimension=dimension,
            turing_dir=str(turing_dir) if turing_dir else None,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]✗ {exc}[/]")
        raise typer.Exit(1)
    console.print(
        f"[green]✓ Done.[/] index [bold]{result['index']}[/]: "
        f"{result['vectors']:,} vectors, dim {result['dimension']}, metric {result['metric']}"
        + (f"  [dim](skipped {result['skipped']})[/]" if result["skipped"] else "")
    )


@app.command()
def status() -> None:
    """Show the managed container's state and connection details."""
    settings = Settings.load()
    state = docker_pg.container_state(settings.container)
    color = {"running": "green", "stopped": "yellow", "absent": "red"}.get(state, "white")
    console.print(f"container [bold]{settings.container}[/]: [{color}]{state}[/]")
    console.print(f"image     {settings.image}")
    console.print(f"database  {settings.dbname}")
    console.print(f"connect   {settings.host}:{settings.port} (user={settings.user})")
    if state == "running":
        try:
            for schema, table, count in docker_pg.table_summary(settings):
                console.print(f"  • {schema}.{table}: {count:,} rows")
        except Exception:  # noqa: BLE001
            pass


@app.command()
def stop(
    remove: bool = typer.Option(False, "--rm", help="Remove the container after stopping."),
) -> None:
    """Stop (and optionally remove) the managed Postgres container."""
    settings = Settings.load()
    state = docker_pg.container_state(settings.container)
    if state == "absent":
        console.print(f"[yellow]No such container:[/] {settings.container}")
        return
    docker_pg.stop_container(settings, remove=remove)
    console.print(f"[green]✓[/] stopped {settings.container}")
    if remove:
        console.print(f"[green]✓[/] removed {settings.container}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
