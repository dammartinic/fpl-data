#!/usr/bin/env python3
"""
Daily Fantasy Premier League snapshot.

Fetches the public FPL API and writes date-partitioned Parquet into ./data,
which GitHub Actions then commits. Databricks reads that repo via a Git folder.

The point of this job is HISTORY. The FPL API is stateless for ownership,
price and transfer flow -- there is no endpoint that tells you what a player's
ownership was three days ago. If nobody snapshots it, it is gone forever.

Env vars:
  FPL_ENTRY_ID   optional. Your FPL manager id (the number in the URL when you
                 view your own team). Enables the my-team tables.
  DATA_DIR       optional, defaults to ./data
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fantasy.premierleague.com/api"

# The FPL API returns 403 to the default python-requests user-agent.
# This is the single most common reason a working script "cannot connect".
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
ENTRY_ID = os.environ.get("FPL_ENTRY_ID", "").strip()

# Columns that FPL returns as strings but which are really numbers.
# Casting here saves a pile of pain in SQL later.
NUMERIC_STRING_COLS = [
    "selected_by_percent", "form", "points_per_game", "value_form", "value_season",
    "ep_this", "ep_next",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "saves_per_90", "starts_per_90", "clean_sheets_per_90",
    "defensive_contribution_per_90",
]


def get(path: str, *, allow_404: bool = False, attempts: int = 4) -> dict | list | None:
    """GET with backoff. Returns None on an allowed 404."""
    url = f"{BASE}/{path.lstrip('/')}"
    last = None
    for i in range(attempts):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 404 and allow_404:
                print(f"  404 (expected) {url}")
                return None
            r.raise_for_status()
            return r.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            wait = 2 ** i
            print(f"  attempt {i + 1}/{attempts} failed for {url}: {exc} -- retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}: {last}")


def write_partition(df: pd.DataFrame, table: str, snapshot_date: str) -> None:
    """Write one date partition. Re-running the same day overwrites cleanly."""
    out_dir = DATA_DIR / table / f"snapshot_date={snapshot_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "part-000.parquet"
    df.to_parquet(path, index=False, compression="snappy")
    print(f"  wrote {len(df):>5} rows -> {path}")


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_STRING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> int:
    now = datetime.now(timezone.utc)
    snapshot_date = now.strftime("%Y-%m-%d")
    snapshot_ts = now.isoformat()
    print(f"FPL snapshot {snapshot_ts}")

    # ---------------------------------------------------------------- bootstrap
    print("bootstrap-static/")
    bootstrap = get("bootstrap-static/")

    # Keep the raw payload gzipped so every parsing mistake is replayable
    # without re-fetching history you can never get back.
    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_dir / f"bootstrap_{snapshot_date}.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(bootstrap, fh)
    print(f"  raw payload saved ({len(bootstrap.get('elements', []))} elements)")

    teams = {t["id"]: t for t in bootstrap["teams"]}
    pos = {p["id"]: p for p in bootstrap["element_types"]}

    events = bootstrap.get("events", [])
    current_gw = next((e["id"] for e in events if e.get("is_current")), None)
    next_gw = next((e["id"] for e in events if e.get("is_next")), None)
    print(f"  current GW={current_gw}  next GW={next_gw}")

    # --- players: take EVERY column FPL gives us, so a new field is never lost
    players = pd.json_normalize(bootstrap["elements"])
    players = coerce_numeric(players)

    # now_cost is in tenths of a million. 155 -> 15.5
    if "now_cost" in players.columns:
        players["price_m"] = players["now_cost"] / 10.0
    for col in ("cost_change_event", "cost_change_start"):
        if col in players.columns:
            players[f"{col}_m"] = players[col] / 10.0

    players["team_name"] = players["team"].map(lambda t: teams.get(t, {}).get("name"))
    players["team_short"] = players["team"].map(lambda t: teams.get(t, {}).get("short_name"))
    players["position"] = players["element_type"].map(
        lambda p: pos.get(p, {}).get("singular_name_short")
    )
    players["net_transfers_event"] = (
        players.get("transfers_in_event", 0) - players.get("transfers_out_event", 0)
    )
    players["snapshot_ts"] = snapshot_ts
    players["current_gw"] = current_gw

    write_partition(players, "player_daily", snapshot_date)

    # --- teams and events, small but useful for joins
    write_partition(pd.json_normalize(bootstrap["teams"]).assign(snapshot_ts=snapshot_ts),
                    "teams_daily", snapshot_date)
    write_partition(pd.json_normalize(events).assign(snapshot_ts=snapshot_ts),
                    "events_daily", snapshot_date)

    # ---------------------------------------------------------------- fixtures
    print("fixtures/")
    fixtures = pd.json_normalize(get("fixtures/"))
    fixtures["snapshot_ts"] = snapshot_ts
    write_partition(fixtures, "fixtures", snapshot_date)

    # ---------------------------------------------------------------- my team
    if ENTRY_ID:
        print(f"entry/{ENTRY_ID}/")
        entry = get(f"entry/{ENTRY_ID}/", allow_404=True)
        if entry:
            write_partition(pd.json_normalize(entry).assign(snapshot_ts=snapshot_ts),
                            "my_entry", snapshot_date)

        history = get(f"entry/{ENTRY_ID}/history/", allow_404=True)
        if history:
            gw_hist = pd.json_normalize(history.get("current", []))
            if not gw_hist.empty:
                gw_hist["entry_id"] = ENTRY_ID
                gw_hist["snapshot_ts"] = snapshot_ts
                # value and bank are also in tenths
                for col in ("value", "bank"):
                    if col in gw_hist.columns:
                        gw_hist[f"{col}_m"] = gw_hist[col] / 10.0
                write_partition(gw_hist, "my_gw_history", snapshot_date)

        # Picks only exist once a gameweek's deadline has passed.
        if current_gw:
            picks = get(f"entry/{ENTRY_ID}/event/{current_gw}/picks/", allow_404=True)
            if picks and picks.get("picks"):
                pk = pd.json_normalize(picks["picks"])
                pk["entry_id"] = ENTRY_ID
                pk["event"] = current_gw
                pk["active_chip"] = picks.get("active_chip")
                pk["snapshot_ts"] = snapshot_ts
                write_partition(pk, "my_picks", snapshot_date)
        else:
            print("  no current gameweek yet -- skipping picks")
         
        # Transfer history. Full log every run -- cheap, and the source
        # is already the complete history, so there's nothing to append to.
        transfers = get(f"entry/{ENTRY_ID}/transfers/", allow_404=True)
        if transfers:
            tr = pd.json_normalize(transfers)
            if not tr.empty:
                tr["entry_id"] = ENTRY_ID
                tr["snapshot_ts"] = snapshot_ts
                # element_in_cost / element_out_cost are in tenths, same as now_cost
                for col in ("element_in_cost", "element_out_cost"):
                    if col in tr.columns:
                        tr[f"{col.replace('_cost', '')}_m"] = tr[col] / 10.0
                write_partition(tr, "my_transfers", snapshot_date)
    else:
        print("FPL_ENTRY_ID not set -- skipping my-team tables")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
