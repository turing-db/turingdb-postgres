# turingdb-postgres

Turn a PostgreSQL dump into a [TuringDB](https://turing.bio) graph.

Point it at a `.dump` file and it restores the dump into a throwaway, dockerised
Postgres (no local `psql`/`pg_restore` required), lets you explore the tables,
then builds a graph — **one node per row, one edge per foreign key** — and, if
your data carries embeddings, a vector index for similarity search.

Everything is available from the `tpg` command line and from a small Python API.

## Requirements

- [`uv`](https://docs.astral.sh/uv/)
- Docker — Postgres runs in a container
- A running TuringDB server — only for the `graph` and `vector` steps

## Install

```bash
uv sync
```

This installs two identical entry points: `tpg` (short) and `turingdb-postgres`.

## Quickstart

```bash
# 1. Restore a dump into a dockerised Postgres (foreign keys are rebuilt for you)
uv run tpg load /path/to/dump.pgdump --recreate

# 2. Look around
uv run tpg show                       # interactive SQL explorer
uv run tpg show --tables              # ...or just list the tables

# 3. Build a TuringDB graph: tables → nodes, foreign keys → edges
uv run tpg graph mygraph --dry-run    # preview the mapping first
uv run tpg graph mygraph

# 4. Done (the restored data stays in the container until you --rm it)
uv run tpg stop
```

## Commands

### `load` — restore a dump

```bash
uv run tpg load /path/to/dump.pgdump --recreate
```

Creates a `postgres:17` container (`turingdb-pg`, on host port `55432`) and runs
`pg_restore` inside it, so you never need Postgres installed locally. The
database name is read from the dump, and the connection details are saved to
`~/.turingdb-postgres/state.json` so every other command can reconnect.

It also **establishes foreign keys** — what the graph step turns into edges.
Since many dumps ship primary keys only, `load` adds:

- any FK you pass explicitly: `--fk orders.customer_id=customers.id` (repeatable), and
- FKs inferred by naming convention (an `<x>_id` column → a table named `<x>`),
  each **checked against the actual rows** first — a relationship the data
  doesn't support is never created. Disable with `--no-infer-fks`.

Other options: `--dbname`, `--port`, `--container`, `--image`, `-j/--jobs`.

### `show` — explore the data

```bash
uv run tpg show                                    # interactive REPL
uv run tpg show --tables                           # list tables and exit
uv run tpg show --table public.customers           # describe + sample a table
uv run tpg show --sql "select count(*) from public.orders"
```

In the REPL the first word is a command; anything else runs as SQL:

| command | what it does |
|---|---|
| `tables` / `schemas` | list tables / schemas |
| `describe <table>` | columns + constraints |
| `sample <table>` | show sample rows |
| `fk <child>.<col> -> <parent>.<col>` | add a foreign key |
| `limit <n>` | row limit for samples and queries |
| `help` / `quit` | help / exit |

### `fk` — add a foreign key

```bash
uv run tpg fk 'orders.customer_id -> customers.id'
uv run tpg fk 'orders.customer_id=customers.id' 'items.order_id=orders.id'   # repeatable
```

Adds a foreign key to the already-loaded database — handy when `load` didn't
infer one, or you skipped inference. It's the same relationship the `graph` step
reads to build an edge, so you can add a missing link and re-run `graph`. The
arrow accepts `->`, `=`, or `:`, and adding an FK that already exists is a no-op.

### `graph` — build the graph

```bash
uv run tpg graph mygraph --dry-run    # preview, import nothing
uv run tpg graph mygraph              # build it
```

The mapping is derived from the schema, so there's nothing to configure:

- **each table → a node label**, PascalCased (`order_items` → `OrderItems`),
  keyed by its single-column primary key, with the scalar columns as properties;
- **each foreign key → an edge**, following the reference direction:
  `orders.customer_id → customers.id` becomes `(Orders)-[:CUSTOMER]->(Customers)`,
  named after the foreign-key column.

Array columns are left out by default (`--include-arrays` to keep them), and
anything that can't be mapped cleanly — composite keys, tables without a primary
key, a column name that carries two different types — is **reported, not
silently dropped**.

Options: `--host` (default `http://localhost:6666`), `--schema` / `--include` /
`--exclude` to scope which tables are used, `--append` to add to an existing
graph instead of clearing it, `--include-arrays`, `--dry-run`.

### `vector` — build a similarity index

If a table holds embeddings, index them for k-nearest-neighbour search:

```bash
uv run tpg vector docs --table documents --key id --column embedding
uv run tpg vector docs --sql "SELECT id, embedding FROM documents"   # any custom query
```

It streams `(id, embedding)` pairs — an **integer** id (so search results join
back to your nodes) and a numeric array — into a `CREATE VECTOR INDEX` +
`LOAD VECTOR`. The dimension is detected automatically and malformed rows are
skipped.

> TuringDB loads the CSV from its own `<turing-dir>/data`, so pass the same
> `--turing-dir` you started the server with (default `~/.turing`) — the file is
> placed in its `data/` subfolder for you.

Options: `--host`, `--graph`, `--metric COSINE|EUCLID`, `--dimension`,
`--turing-dir`, `--limit`.

### `status` / `stop` — manage the container

```bash
uv run tpg status        # container state, connection details, row counts
uv run tpg stop          # stop it (data is kept)
uv run tpg stop --rm     # stop and delete it
```

## Python API

Everything above is also a library. `PostgresDump` returns plain Python data
(lists of dicts, or pandas DataFrames) instead of printing:

```python
from turingdb_postgres import PostgresDump

db = PostgresDump("/path/to/dump.pgdump").load(recreate=True)   # restore + infer FKs
db = PostgresDump.from_state()                                  # ...or reconnect

db.tables()                       # [{'schema', 'table', 'rows', 'total_size'}, ...]
db.describe("customers")          # {'columns': [...], 'constraints': [...]}
db.sample("orders", limit=5)
db.query("SELECT status, count(*) FROM orders GROUP BY 1")
db.dataframe("SELECT * FROM orders")          # pandas
db.add_foreign_key("orders.customer_id -> customers.id")
db.stop()                         # or use `with PostgresDump(...) as db: ...`
```

Build the graph — the same auto-derivation the CLI uses:

```python
db.import_graph_from_schema("mygraph")        # tables → nodes, FKs → edges
```

...or spell the mapping out yourself for full control:

```python
from turingdb_postgres import NodeSpec, EdgeSpec

db.import_graph(
    "mygraph",
    nodes=[
        NodeSpec("Customer", key="id", query="SELECT id, name FROM customers"),
        NodeSpec("Order",    key="id", query="SELECT id, total FROM orders"),
    ],
    edges=[
        EdgeSpec("PLACED", from_label="Customer", to_label="Order",
                 query="SELECT customer_id, id FROM orders"),   # (from-key, to-key)
    ],
)
```

For a parent→child hierarchy imported as a **single commit**, use
`db.import_graph_tree(...)` with `Level(...)` definitions. To index embeddings:

```python
db.build_vector_index("docs", table="documents", key="id", column="embedding")
```

## Project layout

```
src/turingdb_postgres/
  cli.py         # the `tpg` command line (load / show / graph / vector / status / stop)
  api.py         # PostgresDump — the programmatic API
  graph.py       # graph importer, FK-based auto-derivation, vector-index builder
  docker_pg.py   # Docker lifecycle, pg_restore, foreign-key helpers
  explore.py     # interactive + one-shot data exploration
  config.py      # connection settings + saved state
```
