"""Programmatic API for loading and exploring a Postgres dump.

This is the same functionality the CLI exposes, wrapped in a single class that
returns plain Python data (lists of dicts) instead of rendering to a console —
so it can be driven from a script or notebook.

Example
-------
    from turingdb_postgres import PostgresDump

    # Restore a dump into a dockerised Postgres (dbname auto-detected),
    # adding inferred foreign keys.
    db = PostgresDump("/path/to/dump.pgdump").load(recreate=True)

    # ...or connect to an instance a previous `load`/`tpg load` created:
    db = PostgresDump.from_state()

    db.tables()                       # [{'schema': ..., 'table': ..., 'rows': ...}, ...]
    db.describe("customers")          # {'columns': [...], 'constraints': [...]}
    db.sample("orders", limit=5)      # [{...}, ...]
    db.add_foreign_key("orders.customer_id -> customers.id")
    rows = db.query("SELECT status, count(*) FROM public.orders GROUP BY 1")
    db.stop()

`PostgresDump` is also a context manager; leaving the block closes the
connection (it does not stop the container).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import psycopg
from psycopg import sql

from . import docker_pg
from .config import (
    DEFAULT_CONTAINER,
    DEFAULT_DBNAME,
    DEFAULT_HOST,
    DEFAULT_IMAGE,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
    Settings,
)

Row = dict[str, Any]


class PostgresDump:
    """A Postgres dump restored into a dockerised Postgres instance."""

    def __init__(
        self,
        dump: str | Path | None = None,
        *,
        dbname: str | None = None,
        container: str = DEFAULT_CONTAINER,
        image: str = DEFAULT_IMAGE,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        dump:
            Path to the dump file. Required to :meth:`load`; omit it when only
            connecting to an already-loaded instance.
        dbname:
            Target database name. If omitted, :meth:`load` auto-detects it from
            the dump header (falling back to the filename).
        verbose:
            When False, suppresses the progress output the load/FK steps print.
        """
        self._dump = Path(dump).expanduser().resolve() if dump else None
        self._dbname_override = dbname
        self.settings = Settings(
            container=container,
            image=image,
            host=host,
            port=port,
            user=user,
            password=password,
            dbname=dbname or DEFAULT_DBNAME,
        )
        self.verbose = verbose
        self._conn: psycopg.Connection | None = None

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_state(cls, *, verbose: bool = True) -> "PostgresDump":
        """Connect using the settings a previous load persisted to disk."""
        s = Settings.load()
        return cls(
            dbname=s.dbname,
            container=s.container,
            image=s.image,
            host=s.host,
            port=s.port,
            user=s.user,
            password=s.password,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        return (
            f"PostgresDump(dbname={self.settings.dbname!r}, "
            f"container={self.settings.container!r}, "
            f"{self.settings.host}:{self.settings.port}, status={self.status()!r})"
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    @contextmanager
    def _maybe_quiet(self) -> Iterator[None]:
        """Silence docker_pg's progress prints when verbose is False."""
        if self.verbose:
            yield
            return
        from rich.console import Console

        old = docker_pg.console
        docker_pg.console = Console(quiet=True)
        try:
            yield
        finally:
            docker_pg.console = old

    def load(
        self,
        *,
        recreate: bool = False,
        jobs: int = 4,
        foreign_keys: Sequence[str] = (),
        infer_foreign_keys: bool = True,
        save_state: bool = True,
    ) -> "PostgresDump":
        """Spin up Postgres, restore the dump, and establish foreign keys.

        Returns ``self`` so calls can be chained.
        """
        if self._dump is None:
            raise ValueError("no dump path was provided; construct PostgresDump(dump=...)")

        with self._maybe_quiet():
            docker_pg.ensure_docker()
            name, _ = docker_pg.resolve_dbname(self.settings.image, self._dump, self._dbname_override)
            self.settings.dbname = name
            self._reset_connection()

            docker_pg.start_container(self.settings, self._dump.parent)
            docker_pg.wait_ready(self.settings)
            docker_pg.create_database(self.settings, recreate=recreate)
            docker_pg.restore(self.settings, self._dump.name, jobs=jobs)

            specs = [docker_pg.parse_fk_spec(s) for s in foreign_keys]
            if specs:
                docker_pg.apply_foreign_keys(self.settings, specs)
            if infer_foreign_keys:
                docker_pg.apply_foreign_keys(self.settings, docker_pg.infer_foreign_keys(self.settings))

        if save_state:
            self.settings.save()
        return self

    def start(self) -> "PostgresDump":
        """Ensure the container is running (does not restore). Needs the dump
        directory to be mountable if the container must be created."""
        with self._maybe_quiet():
            docker_pg.ensure_docker()
            mount = self._dump.parent if self._dump else Path.cwd()
            docker_pg.start_container(self.settings, mount)
            docker_pg.wait_ready(self.settings)
        return self

    def stop(self, remove: bool = False) -> None:
        """Stop (and optionally remove) the container."""
        self.close()
        docker_pg.stop_container(self.settings, remove=remove)

    def status(self) -> str:
        """'running', 'stopped', or 'absent'."""
        return docker_pg.container_state(self.settings.container)

    def is_running(self) -> bool:
        return self.status() == "running"

    def save_state(self) -> None:
        """Persist connection settings so `from_state`/`tpg show` can reconnect."""
        self.settings.save()

    # ------------------------------------------------------------------ #
    # Connection / querying
    # ------------------------------------------------------------------ #
    def _reset_connection(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    @property
    def connection(self) -> psycopg.Connection:
        """A lazily-opened, reused autocommit connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.settings.conninfo(), autocommit=True)
        return self._conn

    def close(self) -> None:
        """Close the pooled connection (the container keeps running)."""
        self._reset_connection()

    def query(self, statement: str | sql.Composable, params: Sequence[Any] | None = None) -> list[Row]:
        """Run a query and return rows as a list of dicts (empty for non-SELECT)."""
        cur = self.connection.execute(statement, params)
        if cur.description is None:
            return []
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def execute(self, statement: str | sql.Composable, params: Sequence[Any] | None = None) -> str:
        """Run a non-SELECT statement; return the status message (e.g. 'INSERT 0 1')."""
        return self.connection.execute(statement, params).statusmessage or ""

    def dataframe(self, statement: str | sql.Composable, params: Sequence[Any] | None = None):
        """Run a query and return a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.query(statement, params))

    def stream(
        self, statement: str | sql.Composable, params: Sequence[Any] | None = None, *, batch: int = 2000
    ) -> Iterator[Row]:
        """Yield rows one at a time via a server-side cursor.

        Uses its own connection so huge result sets (e.g. a large text column)
        are streamed from Postgres in `batch`-sized chunks rather than buffered
        entirely in memory.
        """
        conn = psycopg.connect(self.settings.conninfo())
        try:
            with conn.cursor(name="tdbpg_stream") as cur:
                cur.itersize = batch
                cur.execute(statement, params)
                cols = [d.name for d in cur.description]
                for row in cur:
                    yield dict(zip(cols, row))
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def schemas(self) -> list[str]:
        rows = self.query(
            """SELECT nspname AS schema FROM pg_namespace
               WHERE nspname NOT IN ('pg_catalog', 'information_schema')
                 AND nspname NOT LIKE 'pg_%' ORDER BY 1"""
        )
        return [r["schema"] for r in rows]

    def tables(self) -> list[Row]:
        """Every user table with estimated row count and on-disk size."""
        return self.query(
            """
            SELECT n.nspname AS schema,
                   c.relname AS table,
                   c.reltuples::bigint AS rows,
                   pg_total_relation_size(c.oid) AS total_bytes,
                   pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size
            FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r'
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY pg_total_relation_size(c.oid) DESC
            """
        )

    def _resolve(self, table: str) -> tuple[str, str]:
        """Resolve a possibly unqualified name to (schema, table)."""
        if "." in table:
            schema, name = table.split(".", 1)
            return schema, name
        rows = self.query(
            """SELECT table_schema FROM information_schema.tables
               WHERE table_name = %s
                 AND table_schema NOT IN ('pg_catalog', 'information_schema')""",
            (table,),
        )
        if not rows:
            raise ValueError(f"no such table: {table}")
        if len(rows) > 1:
            schemas = ", ".join(r["table_schema"] for r in rows)
            raise ValueError(f"table '{table}' is ambiguous ({schemas}); qualify with a schema")
        return rows[0]["table_schema"], table

    def columns(self, table: str) -> list[Row]:
        schema, name = self._resolve(table)
        return self.query(
            """SELECT column_name, data_type, is_nullable, column_default
               FROM information_schema.columns
               WHERE table_schema = %s AND table_name = %s
               ORDER BY ordinal_position""",
            (schema, name),
        )

    def constraints(self, table: str) -> list[Row]:
        schema, name = self._resolve(table)
        return self.query(
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
            (f"{schema}.{name}",),
        )

    def describe(self, table: str) -> dict[str, list[Row]]:
        """Columns + constraints for a table."""
        return {"columns": self.columns(table), "constraints": self.constraints(table)}

    def sample(self, table: str, limit: int = 20) -> list[Row]:
        schema, name = self._resolve(table)
        return self.query(
            sql.SQL("SELECT * FROM {}.{} LIMIT {}").format(
                sql.Identifier(schema), sql.Identifier(name), sql.Literal(limit)
            )
        )

    def count(self, table: str) -> int:
        schema, name = self._resolve(table)
        return self.query(
            sql.SQL("SELECT count(*) AS n FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(name)
            )
        )[0]["n"]

    def foreign_keys(self, table: str | None = None) -> list[Row]:
        """List foreign keys, optionally just for one table."""
        base = """
            SELECT conname AS name,
                   conrelid::regclass::text AS child_table,
                   (SELECT string_agg(a.attname, ',' ORDER BY k.ord)
                      FROM unnest(c.conkey) WITH ORDINALITY AS k(attnum, ord)
                      JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum
                   ) AS child_columns,
                   confrelid::regclass::text AS parent_table,
                   (SELECT string_agg(a.attname, ',' ORDER BY k.ord)
                      FROM unnest(c.confkey) WITH ORDINALITY AS k(attnum, ord)
                      JOIN pg_attribute a ON a.attrelid = c.confrelid AND a.attnum = k.attnum
                   ) AS parent_columns,
                   pg_get_constraintdef(c.oid) AS definition
            FROM pg_constraint c
            WHERE c.contype = 'f'
        """
        if table is None:
            return self.query(base + " ORDER BY child_table, name")
        schema, name = self._resolve(table)
        return self.query(base + " AND c.conrelid = %s::regclass ORDER BY name", (f"{schema}.{name}",))

    # ------------------------------------------------------------------ #
    # Foreign-key management
    # ------------------------------------------------------------------ #
    def add_foreign_key(self, spec: str) -> int:
        """Add a FK from arrow/equals syntax, e.g. 'a.col -> b.col'.

        Returns the number of constraints created (0 if it already existed).
        """
        parsed = docker_pg.parse_fk_spec(spec)
        with self._maybe_quiet():
            return docker_pg.apply_foreign_keys(self.settings, [parsed])

    def add_inferred_foreign_keys(self) -> list[tuple[str, str, str, str]]:
        """Infer FKs by naming convention (validated against the data), add
        them, and return the specs that were considered."""
        with self._maybe_quiet():
            specs = docker_pg.infer_foreign_keys(self.settings)
            docker_pg.apply_foreign_keys(self.settings, specs)
        return specs

    # ------------------------------------------------------------------ #
    # Graph import
    # ------------------------------------------------------------------ #
    def import_graph(
        self,
        graph: str,
        nodes,
        edges=(),
        *,
        host: str = "http://localhost:6666",
        recreate: bool = True,
        **importer_kwargs,
    ) -> dict:
        """Import this database into a TuringDB graph.

        `nodes`/`edges` are NodeSpec/EdgeSpec lists (see turingdb_postgres.graph).
        Returns per-label/per-type counts.
        """
        from .graph import GraphImporter

        importer = GraphImporter(self, graph, host=host, verbose=self.verbose, **importer_kwargs)
        return importer.run(nodes, edges, recreate=recreate)

    def import_graph_tree(
        self,
        graph: str,
        query: str,
        levels,
        *,
        host: str = "http://localhost:6666",
        recreate: bool = True,
        **importer_kwargs,
    ) -> dict:
        """Import a parent→child hierarchy into TuringDB as a SINGLE commit.

        `levels` is a list of turingdb_postgres.graph.Level (root first) and
        `query` is a denormalized SQL query ordered by each level's key.
        """
        from .graph import GraphImporter

        importer = GraphImporter(self, graph, host=host, verbose=self.verbose, **importer_kwargs)
        return importer.import_tree(query, levels, recreate=recreate)

    def import_graph_from_schema(
        self,
        graph: str,
        *,
        host: str = "http://localhost:6666",
        schemas: Sequence[str] | None = None,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        recreate: bool = True,
        include_arrays: bool = False,
        **importer_kwargs,
    ) -> dict:
        """Auto-derive the graph from the schema and import it.

        Nodes come from tables and edges from foreign keys (see
        `turingdb_postgres.graph.derive_specs`). Returns per-label/per-type counts.
        """
        from .graph import derive_specs

        nodes, edges, notes = derive_specs(
            self, schemas=schemas, include=include, exclude=exclude, include_arrays=include_arrays
        )
        if self.verbose:
            for note in notes:
                print(f"• {note}")
        if not nodes:
            raise ValueError("no tables selected to import")
        return self.import_graph(graph, nodes, edges, host=host, recreate=recreate, **importer_kwargs)

    def build_vector_index(
        self,
        index: str,
        *,
        sql: str | None = None,
        table: str | None = None,
        key: str | None = None,
        column: str | None = None,
        host: str = "http://localhost:6666",
        graph: str | None = None,
        metric: str = "COSINE",
        dimension: int | None = None,
        turing_dir: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Build a TuringDB vector index from embeddings stored in Postgres.

        Provide either ``sql`` (returning ``(int id, numeric array)`` rows) or
        the ``table``/``key``/``column`` triple. See
        `turingdb_postgres.graph.build_vector_index` for the mechanics.
        """
        from .graph import build_vector_index as _build

        if sql is None:
            if not (table and key and column):
                raise ValueError("provide sql=..., or table=, key= and column=")
            schema, name = self._resolve(table)
            sql = f'SELECT "{key}", "{column}" FROM "{schema}"."{name}"'
        return _build(
            self,
            index,
            sql,
            host=host,
            graph=graph,
            metric=metric,
            dimension=dimension,
            turing_dir=turing_dir,
            limit=limit,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "PostgresDump":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
