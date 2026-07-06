"""Generic relational → TuringDB graph importer.

The mapping is declarative and data-agnostic: you describe node sources and
edge sources with SQL, and the importer streams rows from Postgres and writes
nodes/edges into a TuringDB graph. Anything domain-specific (which tables map to
which labels, how keys join) lives in the caller's SQL, not here.

    from turingdb_postgres import PostgresDump
    from turingdb_postgres.graph import NodeSpec, EdgeSpec

    db = PostgresDump.from_state()
    db.import_graph(
        "mygraph",
        nodes=[
            NodeSpec("Customer", key="id", query="SELECT id, name FROM public.customers"),
            NodeSpec("Order",    key="id", query="SELECT id, total FROM public.orders"),
        ],
        edges=[
            EdgeSpec("PLACED", from_label="Customer", to_label="Order",
                     query="SELECT customer_id, id FROM public.orders"),
        ],
    )

Verified TuringDB mechanics this relies on:
- writes go through the change workflow (new_change → checkout → ... → CHANGE SUBMIT);
- nodes must be COMMITted before edges can MATCH them;
- string literals use double quotes with backslash escaping; numbers are bare;
- a property name has a single type across the whole graph (so keep types consistent).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .api import PostgresDump

_UNSET = object()


def to_cypher(value: object) -> str:
    """Encode a Python value as a Cypher literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(to_cypher(x) for x in value) + "]"
    s = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{s}"'


@dataclass
class NodeSpec:
    """A source of nodes.

    label:      TuringDB node label.
    key:        property that uniquely identifies the node (indexed, and used
                by edges to MATCH it). Must be one of the query's columns, or
                None for a keyless node (no index; can't be an edge endpoint).
    query:      SQL whose columns become node properties (the SELECT list is
                the property list).
    properties: restrict to these columns (default: all returned columns).
    rename:     map column name → property name.
    """

    label: str
    key: str | None
    query: str
    properties: Sequence[str] | None = None
    rename: dict[str, str] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    """A source of edges: ``(from)-[:type]->(to)``.

    query must return two columns: the from-node's key value and the to-node's
    key value (in that order). Rows with a null on either side are skipped.
    """

    type: str
    from_label: str
    to_label: str
    query: str
    from_key: str = "id"
    to_key: str = "id"


@dataclass
class Level:
    """One tier of a tree import (see GraphImporter.import_tree).

    key:        the (denormalized) query column that identifies a node at this
                tier — used to detect when a new node begins.
    properties: {query column -> node property name} for this tier.
    edge_type:  relationship from the PARENT tier to this node (None for root).
    """

    label: str
    key: str
    properties: dict[str, str] = field(default_factory=dict)
    edge_type: str | None = None

    def key_property(self) -> str:
        return self.properties.get(self.key, self.key)


class GraphImporter:
    def __init__(
        self,
        source: "PostgresDump",
        graph: str,
        *,
        host: str = "http://localhost:6666",
        client=None,
        batch_nodes: int = 500,
        batch_edges: int = 100,
        stream_batch: int = 2000,
        commit_every: int | None = None,
        create_indexes: bool = True,
        verbose: bool = True,
    ) -> None:
        from turingdb import TuringDB

        self.source = source
        self.graph = graph
        self.client = client or TuringDB(host=host)
        self.batch_nodes = batch_nodes
        self.batch_edges = batch_edges
        self.stream_batch = stream_batch
        self.commit_every = commit_every  # COMMIT within the change every N write queries
        self.create_indexes = create_indexes
        self.verbose = verbose
        self._since_commit = 0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _write(self, query: str) -> None:
        """Run a write query, optionally COMMITting within the change periodically.

        Each COMMIT becomes its own entry in the graph's version history, so
        `commit_every` is None by default — flat imports then produce just two
        commits (one after nodes, one at CHANGE SUBMIT). For a *single* commit,
        use import_tree(), which needs no intermediate commits at all.
        """
        self.client.query(query)
        if self.commit_every:
            self._since_commit += 1
            if self._since_commit >= self.commit_every:
                self.client.query("COMMIT")
                self._since_commit = 0

    # ------------------------------------------------------------------ #
    def _setup_graph(self, recreate: bool) -> None:
        from turingdb import TuringDBException

        for op in (self.client.create_graph, self.client.load_graph):
            try:
                op(self.graph)
            except TuringDBException:
                pass  # already exists / already loaded
        self.client.set_graph(self.graph)

        if recreate:
            self._log(f"Clearing existing contents of graph '{self.graph}'…")
            change = self.client.new_change()
            self.client.checkout(change=change)
            self.client.query("MATCH (n) DETACH DELETE n")
            self.client.query("CHANGE SUBMIT")
            self.client.checkout()

    def _create_indexes(self, nodes: Sequence[NodeSpec]) -> None:
        from turingdb import TuringDBException

        for prop in sorted({n.key for n in nodes if n.key}):
            try:
                self.client.query(f"CREATE INDEX idx_{prop} FOR (n) ON n.{prop}")
            except TuringDBException:
                pass  # index already exists (DROP INDEX IF EXISTS isn't supported)
        self.client.query("COMMIT")

    def _node_pattern(self, spec: NodeSpec, row: dict) -> str:
        cols = spec.properties or list(row.keys())
        parts = []
        for col in cols:
            val = row.get(col)
            if val is None:
                continue  # omit null properties entirely
            name = spec.rename.get(col, col)
            parts.append(f"{name}:{to_cypher(val)}")
        return f"(:{spec.label} {{{', '.join(parts)}}})"

    def _create_nodes(self, spec: NodeSpec) -> int:
        buf: list[str] = []
        count = 0
        for row in self.source.stream(spec.query, batch=self.stream_batch):
            buf.append(self._node_pattern(spec, row))
            count += 1
            if len(buf) >= self.batch_nodes:
                self._write("CREATE " + ", ".join(buf))
                buf.clear()
        if buf:
            self._write("CREATE " + ", ".join(buf))
        return count

    def _flush_edges(self, spec: EdgeSpec, pairs: list[tuple]) -> None:
        matches, creates = [], []
        for i, (frm, to) in enumerate(pairs):
            matches.append(f"(f{i}:{spec.from_label} {{{spec.from_key}:{to_cypher(frm)}}})")
            matches.append(f"(t{i}:{spec.to_label} {{{spec.to_key}:{to_cypher(to)}}})")
            creates.append(f"(f{i})-[:{spec.type}]->(t{i})")
        self._write("MATCH " + ", ".join(matches) + " CREATE " + ", ".join(creates))

    def _create_edges(self, spec: EdgeSpec) -> int:
        buf: list[tuple] = []
        count = 0
        for row in self.source.stream(spec.query, batch=self.stream_batch):
            vals = list(row.values())
            frm, to = vals[0], vals[1]
            if frm is None or to is None:
                continue
            buf.append((frm, to))
            count += 1
            if len(buf) >= self.batch_edges:
                self._flush_edges(spec, buf)
                buf.clear()
        if buf:
            self._flush_edges(spec, buf)
        return count

    # ------------------------------------------------------------------ #
    def run(
        self,
        nodes: Sequence[NodeSpec],
        edges: Sequence[EdgeSpec] = (),
        *,
        recreate: bool = True,
    ) -> dict:
        """Import the graph. Returns {'nodes': {label: n}, 'edges': {type: n}}."""
        self._setup_graph(recreate)

        change = self.client.new_change()
        self.client.checkout(change=change)
        self._since_commit = 0

        if self.create_indexes:
            self._create_indexes(nodes)

        stats: dict[str, dict[str, int]] = {"nodes": {}, "edges": {}}
        for spec in nodes:
            n = self._create_nodes(spec)
            stats["nodes"][spec.label] = n
            self._log(f"  nodes  {spec.label}: {n:,}")
        self.client.query("COMMIT")  # nodes must be visible before edges MATCH them
        self._since_commit = 0

        for spec in edges:
            e = self._create_edges(spec)
            stats["edges"][spec.type] = e
            self._log(f"  edges  {spec.type}: {e:,}")

        self.client.query("CHANGE SUBMIT")
        self.client.checkout()
        return stats

    # ------------------------------------------------------------------ #
    # Tree import — a SINGLE commit
    # ------------------------------------------------------------------ #
    def _props(self, level: Level, row: dict) -> str:
        parts = []
        for col, prop in level.properties.items():
            val = row.get(col)
            if val is not None:
                parts.append(f"{prop}:{to_cypher(val)}")
        return (" {" + ", ".join(parts) + "}") if parts else ""

    def _build_subtree(self, rows: list[dict], levels: Sequence[Level], counter: list[int]):
        """Turn the (ordered) rows of one root into CREATE pattern fragments.

        Every non-root node is created inline within its incoming relationship
        pattern, so the whole subtree is one connected CREATE — no MATCH, no
        prior COMMIT needed.
        """
        depth = len(levels)
        ncount: dict[str, int] = defaultdict(int)
        ecount: dict[str, int] = defaultdict(int)
        cur_key: list = [None] * depth
        cur_var: list = [None] * depth

        v = f"n{counter[0]}"; counter[0] += 1  # 'v' + digit is reserved in TuringDB Cypher
        patterns = [f"({v}:{levels[0].label}{self._props(levels[0], rows[0])})"]
        cur_var[0] = v
        ncount[levels[0].label] += 1

        for row in rows:
            changed = False  # once an ancestor changes, all descendants are new
            for d in range(1, depth):
                lv = levels[d]
                key = row.get(lv.key)
                if key is None:  # this row has no node at this tier (or below)
                    for dd in range(d, depth):
                        cur_key[dd] = cur_var[dd] = None
                    break
                if changed or key != cur_key[d] or cur_var[d] is None:
                    v = f"n{counter[0]}"; counter[0] += 1  # 'v' + digit is reserved in TuringDB Cypher
                    patterns.append(
                        f"({cur_var[d - 1]})-[:{lv.edge_type}]->({v}:{lv.label}{self._props(lv, row)})"
                    )
                    ncount[lv.label] += 1
                    ecount[lv.edge_type] += 1
                    cur_key[d] = key
                    cur_var[d] = v
                    changed = True
                    for dd in range(d + 1, depth):
                        cur_key[dd] = cur_var[dd] = None
        return patterns, ncount, ecount

    def import_tree(
        self,
        query: str,
        levels: Sequence[Level],
        *,
        recreate: bool = True,
        nodes_per_query: int = 400,
    ) -> dict:
        """Import a parent→child→… hierarchy in a single commit.

        `query` returns denormalized rows (one per leaf path) ordered by each
        level's key, top-down. `levels[0]` is the root. The whole thing runs in
        one change with one CHANGE SUBMIT — clearing (if recreate), index
        creation, and all writes share that single commit.
        """
        from turingdb import TuringDBException

        for op in (self.client.create_graph, self.client.load_graph):
            try:
                op(self.graph)
            except TuringDBException:
                pass
        self.client.set_graph(self.graph)

        change = self.client.new_change()
        self.client.checkout(change=change)
        if recreate:
            self.client.query("MATCH (n) DETACH DELETE n")
        if self.create_indexes:
            for prop in sorted({lv.key_property() for lv in levels}):
                try:
                    self.client.query(f"CREATE INDEX idx_{prop} FOR (n) ON n.{prop}")
                except TuringDBException:
                    pass

        ncount: dict[str, int] = defaultdict(int)
        ecount: dict[str, int] = defaultdict(int)
        patterns: list[str] = []
        counter = [0]
        root_key = levels[0].key

        def flush():
            if patterns:
                self.client.query("CREATE " + ", ".join(patterns))
                patterns.clear()
                counter[0] = 0  # variable names only need to be unique within a query

        def emit(rows: list[dict]):
            p, nc, ec = self._build_subtree(rows, levels, counter)
            patterns.extend(p)
            for k, n in nc.items():
                ncount[k] += n
            for k, n in ec.items():
                ecount[k] += n
            if len(patterns) >= nodes_per_query:
                flush()

        cur_root = _UNSET
        cur_rows: list[dict] = []
        for row in self.source.stream(query, batch=self.stream_batch):
            rk = row.get(root_key)
            if cur_root is _UNSET or rk != cur_root:
                if cur_rows:
                    emit(cur_rows)
                cur_rows = [row]
                cur_root = rk
            else:
                cur_rows.append(row)
        if cur_rows:
            emit(cur_rows)
        flush()

        self.client.query("CHANGE SUBMIT")
        self.client.checkout()

        stats = {"nodes": dict(ncount), "edges": dict(ecount)}
        for label, n in stats["nodes"].items():
            self._log(f"  nodes  {label}: {n:,}")
        for etype, n in stats["edges"].items():
            self._log(f"  edges  {etype}: {n:,}")
        return stats


# --------------------------------------------------------------------------- #
# Auto-derive a graph mapping from the relational schema
#
# `tpg graph` builds a graph without hand-written specs: one node label per
# table, one edge per foreign key. The mapping is mechanical and documented so
# it's predictable — domain-specific mappings still use NodeSpec/EdgeSpec.
# --------------------------------------------------------------------------- #
def _pascal(name: str) -> str:
    """table name -> Label, e.g. 'federal_articles' -> 'FederalArticles'."""
    parts = re.split(r"[^0-9A-Za-z]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or name


def _qi(ident: str) -> str:
    """Quote a SQL identifier."""
    return '"' + ident.replace('"', '""') + '"'


def _qt(schema: str, table: str) -> str:
    return f"{_qi(schema)}.{_qi(table)}"


def _rel_type(fk_col: str, parent_label: str) -> str:
    """Relationship type from an FK column, e.g. 'law_id' -> 'LAW'.

    Falls back to the parent label when the column is just 'id'.
    """
    base = re.sub(r"_(id|fk|key)$", "", fk_col, flags=re.IGNORECASE)
    if base.lower() == "id":
        base = ""
    base = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_")
    return base.upper() if base else parent_label.upper()


_TABLES_SQL = """
SELECT n.nspname AS schema, c.relname AS "table"
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind = 'r'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY 1, 2
"""

_COLUMNS_SQL = """
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
ORDER BY table_schema, table_name, ordinal_position
"""

_PK_SQL = """
SELECT n.nspname AS schema, c.relname AS "table",
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum) AS cols
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE con.contype = 'p'
  AND n.nspname NOT IN ('pg_catalog', 'information_schema')
"""

_FK_SQL = """
SELECT cn.nspname AS child_schema, cc.relname AS child_table,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum) AS child_cols,
       pn.nspname AS parent_schema, pc.relname AS parent_table,
       (SELECT array_agg(a.attname ORDER BY k.ord)
          FROM unnest(con.confkey) WITH ORDINALITY k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum) AS parent_cols
FROM pg_constraint con
JOIN pg_class cc ON cc.oid = con.conrelid
JOIN pg_namespace cn ON cn.oid = cc.relnamespace
JOIN pg_class pc ON pc.oid = con.confrelid
JOIN pg_namespace pn ON pn.oid = pc.relnamespace
WHERE con.contype = 'f'
  AND cn.nspname NOT IN ('pg_catalog', 'information_schema')
ORDER BY child_schema, child_table, con.conname
"""


def derive_specs(
    db: "PostgresDump",
    *,
    schemas: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    include_arrays: bool = False,
) -> tuple[list[NodeSpec], list[EdgeSpec], list[str]]:
    """Derive (nodes, edges, notes) from the live schema.

    - One NodeSpec per table: label = PascalCase(table), key = its single-column
      primary key (None if composite/absent), properties = the scalar columns
      (array columns are skipped unless ``include_arrays``).
    - One EdgeSpec per foreign key, following the reference direction
      ``(child)-[:REL]->(parent)`` with REL derived from the FK column name.

    ``notes`` is a list of human-readable messages about anything skipped or
    risky (composite keys, keyless tables, array columns, property type clashes)
    — surfaced rather than silently dropped.
    """
    inc = {s.lower() for s in include} if include else None
    exc = {s.lower() for s in exclude} if exclude else set()
    sch = {s.lower() for s in schemas} if schemas else None

    def selected(schema: str, table: str) -> bool:
        if sch is not None and schema.lower() not in sch:
            return False
        names = {table.lower(), f"{schema}.{table}".lower()}
        if inc is not None and not (names & inc):
            return False
        if names & exc:
            return False
        return True

    tables = [(r["schema"], r["table"]) for r in db.query(_TABLES_SQL)]
    tables = [(s, t) for (s, t) in tables if selected(s, t)]
    tset = set(tables)

    cols_by: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for r in db.query(_COLUMNS_SQL):
        st = (r["table_schema"], r["table_name"])
        if st in tset:
            cols_by.setdefault(st, []).append((r["column_name"], r["data_type"]))

    pk_by: dict[tuple[str, str], list[str]] = {}
    for r in db.query(_PK_SQL):
        pk_by[(r["schema"], r["table"])] = list(r["cols"] or [])

    # Labels — disambiguate same-named tables in different schemas.
    label_counts: dict[str, int] = defaultdict(int)
    for _s, t in tables:
        label_counts[_pascal(t)] += 1
    label_of: dict[tuple[str, str], str] = {}
    for s, t in tables:
        base = _pascal(t)
        label_of[(s, t)] = (_pascal(s) + base) if label_counts[base] > 1 else base

    def prop_cols(st: tuple[str, str]) -> list[str]:
        return [
            name
            for (name, dt) in cols_by.get(st, [])
            if include_arrays or dt != "ARRAY"
        ]

    notes: list[str] = []
    nodes: list[NodeSpec] = []
    for s, t in tables:
        st = (s, t)
        pk = pk_by.get(st, [])
        key = pk[0] if len(pk) == 1 else None
        pcols = prop_cols(st)
        if not pcols:
            notes.append(f"{s}.{t}: no scalar columns to import — skipped")
            continue
        if key is None:
            notes.append(
                f"{s}.{t}: no single-column primary key — imported as keyless nodes "
                "(no index; cannot be an edge endpoint)"
            )
        query = f"SELECT {', '.join(_qi(c) for c in pcols)} FROM {_qt(s, t)}"
        nodes.append(NodeSpec(label=label_of[st], key=key, query=query, properties=pcols))
        arrays = [name for (name, dt) in cols_by.get(st, []) if dt == "ARRAY"]
        if arrays and not include_arrays:
            notes.append(
                f"{s}.{t}: skipped array column(s) {', '.join(arrays)} "
                "(use --include-arrays to keep them)"
            )

    # Warn about property names that carry more than one type across the graph
    # (TuringDB requires a single type per property name).
    type_map: dict[str, set[str]] = defaultdict(set)
    for st in tset:
        for name, dt in cols_by.get(st, []):
            if include_arrays or dt != "ARRAY":
                type_map[name].add(dt)
    for name, types in sorted(type_map.items()):
        if len(types) > 1:
            notes.append(
                f"property '{name}' appears with differing types ({', '.join(sorted(types))}) — "
                "TuringDB requires one type per property name across the graph"
            )

    edges: list[EdgeSpec] = []
    for r in db.query(_FK_SQL):
        cst = (r["child_schema"], r["child_table"])
        pst = (r["parent_schema"], r["parent_table"])
        ccols = list(r["child_cols"] or [])
        pcols_ = list(r["parent_cols"] or [])
        if cst not in tset or pst not in tset:
            continue  # an endpoint was filtered out
        arrow = (
            f"{cst[0]}.{cst[1]}({','.join(ccols)}) -> "
            f"{pst[0]}.{pst[1]}({','.join(pcols_)})"
        )
        if len(ccols) != 1 or len(pcols_) != 1:
            notes.append(f"FK {arrow}: composite key not supported — edge skipped")
            continue
        child_pk_list = pk_by.get(cst, [])
        if len(child_pk_list) != 1:
            notes.append(
                f"FK {arrow}: child {cst[0]}.{cst[1]} has no single-column primary key — edge skipped"
            )
            continue
        child_pk, fk_col, parent_col = child_pk_list[0], ccols[0], pcols_[0]
        if parent_col not in prop_cols(pst):
            notes.append(f"FK {arrow}: parent key '{parent_col}' is not an imported property — edge skipped")
            continue
        rel = _rel_type(fk_col, label_of[pst])
        query = (
            f"SELECT {_qi(child_pk)}, {_qi(fk_col)} FROM {_qt(*cst)} "
            f"WHERE {_qi(fk_col)} IS NOT NULL"
        )
        edges.append(
            EdgeSpec(
                type=rel,
                from_label=label_of[cst],
                to_label=label_of[pst],
                query=query,
                from_key=child_pk,
                to_key=parent_col,
            )
        )

    return nodes, edges, notes


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #
def build_vector_index(
    source: "PostgresDump",
    index: str,
    sql: str,
    *,
    host: str = "http://localhost:6666",
    graph: str | None = None,
    metric: str = "COSINE",
    dimension: int | None = None,
    turing_dir: str | None = None,
    limit: int | None = None,
    stream_batch: int = 500,
    verbose: bool = True,
) -> dict:
    """Build a TuringDB vector index from embeddings stored in Postgres.

    ``sql`` must return ``(id, embedding)`` rows — an integer id (matching the
    node key used to join search results back to the graph) and a numeric array.
    Rows whose vector is null or the wrong width are skipped.

    The only way to populate an index is ``LOAD VECTOR FROM "<file>"``, which
    the server resolves under ``<turing-dir>/data``. So pass the same
    ``turing_dir`` the TuringDB server was started with (``-turing-dir``, default
    ``~/.turing``); the CSV is written to its ``data`` subdirectory.
    """
    from pathlib import Path

    from turingdb import TuringDB, TuringDBException

    metric = metric.upper()
    if metric not in ("COSINE", "EUCLID"):
        raise ValueError(f"metric must be COSINE or EUCLID, got {metric!r}")

    base = Path(turing_dir).expanduser() if turing_dir else Path.home() / ".turing"
    target_dir = base / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{index}.csv"

    query = sql + (f" LIMIT {int(limit)}" if limit else "")
    dim = dimension
    n = skipped = 0

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"Writing vectors to {csv_path}…")
    with open(csv_path, "w") as f:
        for row in source.stream(query, batch=stream_batch):
            vals = list(row.values())
            if len(vals) < 2:
                raise ValueError("vector SQL must return at least (id, embedding)")
            key, vec = vals[0], vals[1]
            if vec is None:
                skipped += 1
                continue
            vec = list(vec)
            if dim is None:
                dim = len(vec)
            if len(vec) != dim:
                skipped += 1  # wrong-width vectors would make LOAD VECTOR choke
                continue
            try:
                kid = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    f"vector id must be an integer (it joins to a node key), got {key!r}"
                )
            f.write(f"{kid}," + ",".join(repr(x) for x in vec) + "\n")
            n += 1
            if verbose and n % 5000 == 0:
                log(f"  …{n:,} vectors written")

    if n == 0:
        raise ValueError("no vectors to index (every row was null, empty, or the wrong width)")
    log(f"  wrote {n:,} vectors (dim {dim})" + (f", skipped {skipped}" if skipped else ""))

    client = TuringDB(host=host)
    if graph:
        client.set_graph(graph)
    try:
        client.query(f"DELETE VECTOR INDEX {index}")
    except TuringDBException:
        pass  # didn't exist yet
    client.query(f"CREATE VECTOR INDEX {index} WITH DIMENSION {dim} METRIC {metric}")
    log(f"  loading {csv_path.name} into index '{index}' (this can take a while)…")
    client.query(f'LOAD VECTOR FROM "{csv_path.name}" IN {index}')
    log(f"[done] index '{index}': {n:,} vectors, dim {dim}, metric {metric}")

    return {
        "index": index,
        "vectors": n,
        "skipped": skipped,
        "dimension": dim,
        "metric": metric,
        "csv": str(csv_path),
    }
