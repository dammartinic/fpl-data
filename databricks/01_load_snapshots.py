# Databricks notebook source
# MAGIC %md
# MAGIC # FPL snapshots -> Delta
# MAGIC
# MAGIC Reads the Parquet written by the GitHub Action (synced into this workspace
# MAGIC via a Databricks **Git folder**) and lands it in Unity Catalog as Delta.
# MAGIC
# MAGIC No outbound internet is needed — that is the whole point of this design.
# MAGIC The Git folder sync is the network call, and GitHub is reachable from
# MAGIC Free Edition where `fantasy.premierleague.com` is not.
# MAGIC
# MAGIC Schedule this as a Databricks job **once a day, a couple of hours after
# MAGIC the Action runs**. One job, one run per day — Free Edition has a daily
# MAGIC compute quota and blowing it shuts your warehouse down until tomorrow.

# COMMAND ----------

CATALOG = "workspace"      # Free Edition default catalog
SCHEMA = "fpl"

# Path to this Git folder in the workspace. Adjust <you@example.com> and the
# repo name. Easiest way to get it right: right-click the repo in the sidebar
# and choose "Copy path", then strip the trailing notebook name.
REPO_PATH = "/Workspace/Users/<you@example.com>/fpl-data"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")
print(f"writing into {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load
# MAGIC
# MAGIC Read with pandas rather than Spark. The data is tiny (~700 rows a day) and
# MAGIC pandas reads workspace files without the path quirks Spark has under
# MAGIC `/Workspace`. If you would rather use Spark, the equivalent is
# MAGIC `spark.read.option("mergeSchema","true").parquet(f"file:{REPO_PATH}/data/player_daily")`.

# COMMAND ----------

import glob
import os

import pandas as pd


def load_table(name: str):
    """Read every date partition of one table into a single Spark DataFrame."""
    root = os.path.join(REPO_PATH, "data", name)
    files = sorted(glob.glob(os.path.join(root, "snapshot_date=*", "*.parquet")))
    if not files:
        print(f"  {name}: no partitions found under {root}")
        return None

    frames = []
    for f in files:
        # snapshot_date lives in the directory name, not the file
        snapshot_date = os.path.basename(os.path.dirname(f)).split("=", 1)[1]
        part = pd.read_parquet(f)
        part["snapshot_date"] = snapshot_date
        frames.append(part)

    # sort=False keeps column order stable; new FPL fields simply append
    pdf = pd.concat(frames, ignore_index=True, sort=False)
    pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"]).dt.date

    # Parquet round-trips object columns that are all-null as float; make them
    # strings so Spark does not choke on the schema.
    for col in pdf.columns:
        if pdf[col].dtype == "object":
            pdf[col] = pdf[col].astype("string")

    print(f"  {name}: {len(files)} partitions, {len(pdf):,} rows")
    return spark.createDataFrame(pdf)


for table in ["player_daily", "fixtures", "teams_daily", "events_daily",
              "my_entry", "my_gw_history", "my_picks"]:
    sdf = load_table(table)
    if sdf is not None:
        (sdf.write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"bronze_{table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver: the table everything else is built on
# MAGIC
# MAGIC One row per player per day, with day-over-day deltas already computed.
# MAGIC Every "mover" metric on the dashboard falls out of these window functions.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver_player_daily AS
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     snapshot_date,
# MAGIC     id                          AS player_id,
# MAGIC     web_name,
# MAGIC     team_short,
# MAGIC     team_name,
# MAGIC     position,
# MAGIC     price_m,
# MAGIC     selected_by_percent         AS ownership_pct,
# MAGIC     net_transfers_event,
# MAGIC     transfers_in_event,
# MAGIC     transfers_out_event,
# MAGIC     total_points,
# MAGIC     event_points,
# MAGIC     form,
# MAGIC     minutes,
# MAGIC     starts,
# MAGIC     expected_goal_involvements_per_90 AS xgi_per_90,
# MAGIC     expected_goals,
# MAGIC     expected_assists,
# MAGIC     status,
# MAGIC     news,
# MAGIC     chance_of_playing_next_round,
# MAGIC     penalties_order,
# MAGIC     corners_and_indirect_freekicks_order,
# MAGIC     direct_freekicks_order
# MAGIC   FROM bronze_player_daily
# MAGIC )
# MAGIC SELECT
# MAGIC   b.*,
# MAGIC   LAG(ownership_pct) OVER w                      AS ownership_pct_prev,
# MAGIC   ownership_pct - LAG(ownership_pct) OVER w      AS ownership_chg_1d,
# MAGIC   ownership_pct - LAG(ownership_pct, 3) OVER w   AS ownership_chg_3d,
# MAGIC   ownership_pct - LAG(ownership_pct, 7) OVER w   AS ownership_chg_7d,
# MAGIC   price_m - LAG(price_m) OVER w                  AS price_chg_1d,
# MAGIC   price_m - FIRST_VALUE(price_m) OVER (
# MAGIC     PARTITION BY player_id ORDER BY snapshot_date
# MAGIC     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
# MAGIC   )                                              AS price_chg_since_start,
# MAGIC   CASE WHEN total_points > 0 AND price_m > 0
# MAGIC        THEN total_points / price_m END           AS points_per_million
# MAGIC FROM base b
# MAGIC WINDOW w AS (PARTITION BY player_id ORDER BY snapshot_date);

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold: the three dashboard feeds
# MAGIC
# MAGIC Edit `gold_watchlist` to match whoever you are tracking. Everything else
# MAGIC is derived.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Your 15. Edit freely; this is the only hand-maintained table.
# MAGIC CREATE OR REPLACE TABLE gold_my_squad (web_name STRING, role STRING);
# MAGIC INSERT INTO gold_my_squad VALUES
# MAGIC   ('Verbruggen','XI'), ('Gabriel','XI'),   ('Maguire','XI'),
# MAGIC   ('Vuskovic','XI'),   ('B.Fernandes','XI'), ('Mbeumo','XI'),
# MAGIC   ('Ødegaard','XI'),   ('Gross','XI'),     ('Sangaré','XI'),
# MAGIC   ('Haaland','XI'),    ('João Pedro','XI'),
# MAGIC   ('Dúbravka','Bench'),('Robinson','Bench'),
# MAGIC   ('Egeli','Bench'),   ('van Ewijk','Bench');
# MAGIC
# MAGIC -- NOTE: web_name spellings must match the FPL API exactly. Run
# MAGIC --   SELECT DISTINCT web_name FROM silver_player_daily ORDER BY 1
# MAGIC -- and fix any that do not join. Matching on player_id is more robust
# MAGIC -- once you have looked the ids up.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Page 3: biggest movers, EXCLUDING players you already own.
# MAGIC -- The exclusion is the point -- you want to see what is coming.
# MAGIC CREATE OR REPLACE VIEW gold_movers AS
# MAGIC SELECT
# MAGIC   p.snapshot_date, p.web_name, p.team_short, p.position,
# MAGIC   p.price_m, p.ownership_pct,
# MAGIC   p.ownership_chg_1d, p.ownership_chg_3d, p.ownership_chg_7d,
# MAGIC   p.net_transfers_event, p.form, p.xgi_per_90, p.status,
# MAGIC   CASE WHEN p.ownership_pct > 0
# MAGIC        THEN p.net_transfers_event / (p.ownership_pct * 100000)
# MAGIC   END AS transfer_pressure   -- proxy for how close a price change is
# MAGIC FROM silver_player_daily p
# MAGIC LEFT ANTI JOIN gold_my_squad s ON p.web_name = s.web_name
# MAGIC WHERE p.snapshot_date = (SELECT MAX(snapshot_date) FROM silver_player_daily)
# MAGIC   AND p.minutes > 0;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Page 1: your squad, most recent snapshot plus trend columns.
# MAGIC CREATE OR REPLACE VIEW gold_my_squad_latest AS
# MAGIC SELECT s.role, p.*
# MAGIC FROM silver_player_daily p
# MAGIC JOIN gold_my_squad s ON p.web_name = s.web_name
# MAGIC WHERE p.snapshot_date = (SELECT MAX(snapshot_date) FROM silver_player_daily);

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Differential finder: strong value, low ownership. Plot as a scatter of
# MAGIC -- points_per_million (y) against ownership_pct (x) and look top-left.
# MAGIC CREATE OR REPLACE VIEW gold_value_vs_ownership AS
# MAGIC SELECT web_name, team_short, position, price_m, ownership_pct,
# MAGIC        total_points, points_per_million, xgi_per_90, minutes
# MAGIC FROM silver_player_daily
# MAGIC WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM silver_player_daily)
# MAGIC   AND minutes >= 180;

# COMMAND ----------

# MAGIC %md
# MAGIC Now build the dashboard: **New > Dashboard**, add a dataset per view above,
# MAGIC and put each on its own page. Three pages in one dashboard beats three
# MAGIC dashboards — the filters carry across and there is a third as much to maintain.
