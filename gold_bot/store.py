"""Where everything that must survive a restart is kept: sidebar settings, saved strategies, the live session and its trade log.

Default is plain files under data/ (works anywhere, but a host with a temporary disk such as Render's free plan loses them
on every restart or redeploy). Set the environment variable MONGODB_URI (for example a free MongoDB Atlas cluster) and the
same data is kept in MongoDB instead, so it survives. MONGODB_DB names the database (default "gold_bot").
If MongoDB is configured but cannot be reached, the files are used and `error` says why.
"""
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from .data import ROOT

DATA = ROOT / "data"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe(name):
    return "".join(ch for ch in name if ch.isalnum() or ch in " -_").strip() or "strategy"


def _plain(v):
    """JSON / BSON friendly copy (numpy numbers and pandas timestamps become plain Python)."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


class FileStore:
    name = "files (data/ folder)"

    def __init__(self, root=DATA):
        self.root = root
        self.live, self.strats = root / "live", root / "strategies"
        self.live.mkdir(parents=True, exist_ok=True)
        self.strats.mkdir(parents=True, exist_ok=True)

    # settings
    def load_settings(self):
        f = self.root / "settings.json"
        return json.loads(f.read_text()) if f.exists() else None

    def save_settings(self, d):
        (self.root / "settings.json").write_text(json.dumps(_plain({**d, "saved": _now()}), indent=1))

    # strategies
    def list_strategies(self):
        return sorted(p.stem for p in self.strats.glob("*.json"))

    def save_strategy(self, name, doc):
        name = _safe(name)
        (self.strats / f"{name}.json").write_text(json.dumps(_plain({**doc, "name": name, "saved": _now()}), indent=1))
        return name

    def load_strategy(self, name):
        return json.loads((self.strats / f"{name}.json").read_text())

    # live session
    def get_session(self):
        f = self.live / "session.json"
        return json.loads(f.read_text()) if f.exists() else None

    def set_session(self, d):
        (self.live / "session.json").write_text(json.dumps(_plain(d), indent=1))

    def archive_session(self):
        f = self.live / "session.json"
        if f.exists():
            f.rename(self.live / f"session_stopped_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json")

    # trade log
    def event_keys(self):
        f = self.live / "events.csv"
        return set(pd.read_csv(f)["key"]) if f.exists() else set()

    def add_events(self, rows):
        f = self.live / "events.csv"
        pd.DataFrame(_plain(rows)).to_csv(f, mode="a", header=not f.exists(), index=False)


class MongoStore:
    def __init__(self, db):
        self.db = db
        self.name = "MongoDB"

    def load_settings(self):
        d = self.db.settings.find_one({"_id": "app"})
        return {k: v for k, v in d.items() if k != "_id"} if d else None

    def save_settings(self, d):
        self.db.settings.replace_one({"_id": "app"}, _plain({**d, "saved": _now()}), upsert=True)

    def list_strategies(self):
        return sorted(d["_id"] for d in self.db.strategies.find({}, {"_id": 1}))

    def save_strategy(self, name, doc):
        name = _safe(name)
        self.db.strategies.replace_one({"_id": name}, _plain({**doc, "name": name, "saved": _now()}), upsert=True)
        return name

    def load_strategy(self, name):
        d = self.db.strategies.find_one({"_id": name})
        if d is None:
            raise FileNotFoundError(name)
        return {k: v for k, v in d.items() if k != "_id"}

    def get_session(self):
        d = self.db.session.find_one({"_id": "current"})
        return {k: v for k, v in d.items() if k != "_id"} if d else None

    def set_session(self, d):
        self.db.session.replace_one({"_id": "current"}, _plain(d), upsert=True)

    def archive_session(self):
        d = self.db.session.find_one({"_id": "current"})
        if d:
            d.pop("_id")
            self.db.sessions_stopped.insert_one({**d, "stopped": _now()})
            self.db.session.delete_one({"_id": "current"})

    def event_keys(self):
        return {d["_id"] for d in self.db.events.find({}, {"_id": 1})}

    def add_events(self, rows):
        for r in _plain(rows):
            k = r.pop("key")
            self.db.events.replace_one({"_id": k}, r, upsert=True)


def _connect():
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        return FileStore(), ""
    try:
        from pymongo import MongoClient
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        return MongoStore(client[os.environ.get("MONGODB_DB", "gold_bot")]), ""
    except Exception as e:
        err = re.sub(r"//[^@/\s]+@", "//***@", f"{type(e).__name__}: {e}")  # never show the password
        return FileStore(), f"MongoDB is configured but not reachable, so files are used instead. {err[:200]}"


store, error = _connect()

# the functions the rest of the app calls
load_settings, save_settings = store.load_settings, store.save_settings
list_strategies, save_strategy, load_strategy = store.list_strategies, store.save_strategy, store.load_strategy
get_session, set_session, archive_session = store.get_session, store.set_session, store.archive_session
event_keys, add_events = store.event_keys, store.add_events
