"""Connection settings and persisted state shared across subcommands.

`load` writes the connection details it used to a small state file so that
`show` (and, later, the graph-import command) can reconnect without the user
re-typing everything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Where we stash the last-used connection details.
STATE_DIR = Path.home() / ".turingdb-postgres"
STATE_FILE = STATE_DIR / "state.json"

# Defaults.
DEFAULT_CONTAINER = "turingdb-pg"
DEFAULT_IMAGE = "postgres:17"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55432  # host port; avoids clashing with a native postgres on 5432
DEFAULT_USER = "postgres"
DEFAULT_PASSWORD = "postgres"
# `load` resolves the real database name from the dump (see docker_pg.resolve_dbname);
# this is only a last-resort fallback for `show`/`status` if no state file exists yet.
DEFAULT_DBNAME = "postgres"


@dataclass
class Settings:
    """Everything needed to talk to the dockerised Postgres instance."""

    container: str = DEFAULT_CONTAINER
    image: str = DEFAULT_IMAGE
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    user: str = DEFAULT_USER
    password: str = DEFAULT_PASSWORD
    dbname: str = DEFAULT_DBNAME

    def conninfo(self, dbname: str | None = None) -> str:
        """A libpq connection string for psycopg."""
        db = dbname or self.dbname
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={db}"
        )

    def save(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Settings":
        """Load persisted settings, falling back to defaults for missing keys."""
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        return cls()
