"""
db.py — SQLite data layer.

All database access goes through this module. Schema creation is idempotent
(CREATE TABLE IF NOT EXISTS), so it's safe to call DB() on every startup.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS bar_data (
    id         INTEGER PRIMARY KEY,
    symbol     TEXT NOT NULL,
    ts         TEXT NOT NULL,
    open       REAL NOT NULL,
    high       REAL NOT NULL,
    low        REAL NOT NULL,
    close      REAL NOT NULL,
    volume     REAL NOT NULL,
    interval   TEXT NOT NULL DEFAULT '1d',
    fetched_at TEXT NOT NULL,
    UNIQUE(symbol, ts, interval)
);
CREATE INDEX IF NOT EXISTS bar_data_symbol_ts ON bar_data(symbol, ts);

CREATE TABLE IF NOT EXISTS features (
    id            INTEGER PRIMARY KEY,
    symbol        TEXT NOT NULL,
    ts            TEXT NOT NULL,
    feature_set   TEXT NOT NULL DEFAULT 'daily_v1',
    features_json TEXT NOT NULL,
    computed_at   TEXT NOT NULL,
    UNIQUE(symbol, ts, feature_set)
);
CREATE INDEX IF NOT EXISTS features_symbol_ts ON features(symbol, ts, feature_set);

CREATE TABLE IF NOT EXISTS model_registry (
    id               INTEGER PRIMARY KEY,
    model_key        TEXT    NOT NULL,
    version          INTEGER NOT NULL,
    artifact_path    TEXT    NOT NULL,
    feature_contract TEXT    NOT NULL,
    feature_set_name TEXT    NOT NULL DEFAULT 'daily_v1',
    trained_on       TEXT    NOT NULL,
    train_start      TEXT    NOT NULL,
    train_end        TEXT    NOT NULL,
    train_samples    INTEGER,
    test_samples     INTEGER,
    train_accuracy   REAL,
    test_accuracy    REAL,
    test_f1          REAL,
    label_map        TEXT    NOT NULL DEFAULT '{"0":"SELL","1":"HOLD","2":"BUY"}',
    trained_at       TEXT    NOT NULL,
    is_active        INTEGER NOT NULL DEFAULT 1,
    UNIQUE(model_key, version)
);
CREATE INDEX IF NOT EXISTS model_registry_active ON model_registry(model_key, is_active);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id                   INTEGER PRIMARY KEY,
    run_id               TEXT NOT NULL,
    symbol               TEXT NOT NULL,
    strategy             TEXT NOT NULL,
    model_version        INTEGER,
    data_start           TEXT NOT NULL,
    data_end             TEXT NOT NULL,
    start_cash           REAL NOT NULL,
    final_equity         REAL,
    total_return_pct     REAL,
    daily_sharpe         REAL,
    daily_sortino        REAL,
    max_drawdown_pct     REAL,
    n_round_trades       INTEGER,
    hit_rate             REAL,
    profit_factor        REAL,
    commission_per_share REAL,
    slippage_bps         REAL,
    stop_loss_pct        REAL,
    holding_period       INTEGER,
    ran_at               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS backtest_runs_symbol_strategy ON backtest_runs(symbol, strategy, ran_at);
CREATE INDEX IF NOT EXISTS backtest_runs_run_id ON backtest_runs(run_id);

CREATE TABLE IF NOT EXISTS trade_log (
    id              INTEGER PRIMARY KEY,
    backtest_run_id INTEGER NOT NULL REFERENCES backtest_runs(id),
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    ts              TEXT NOT NULL,
    side            TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
    shares          INTEGER NOT NULL,
    fill_price      REAL NOT NULL,
    commission      REAL NOT NULL DEFAULT 0,
    spread_cost     REAL NOT NULL DEFAULT 0,
    exit_reason     TEXT
);
CREATE INDEX IF NOT EXISTS trade_log_run ON trade_log(backtest_run_id);
CREATE INDEX IF NOT EXISTS trade_log_symbol ON trade_log(symbol, ts);

CREATE TABLE IF NOT EXISTS daily_predictions (
    id                  INTEGER PRIMARY KEY,
    symbol              TEXT NOT NULL,
    model_key           TEXT NOT NULL,
    model_version       INTEGER NOT NULL,
    prediction_date     TEXT NOT NULL,
    signal              TEXT NOT NULL CHECK(signal IN ('BUY','SELL','HOLD')),
    confidence          REAL NOT NULL,
    price_at_prediction REAL,
    predicted_at        TEXT NOT NULL,
    UNIQUE(symbol, model_key, prediction_date)
);
CREATE INDEX IF NOT EXISTS daily_predictions_date ON daily_predictions(prediction_date, symbol);
"""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class DB:
    """SQLite data layer. One instance per process; thread-safe via lock."""

    def __init__(self, path: str = "data/trading_sim.db") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._lock, self._connect() as con:
            con.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Bar data
    # ------------------------------------------------------------------

    def upsert_bars(self, symbol: str, interval: str, df: pd.DataFrame) -> int:
        """Insert or replace OHLCV bars. Returns number of rows written."""
        now = _now_utc()
        rows = []
        for ts, row in df.iterrows():
            ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            rows.append((
                symbol,
                ts_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                interval,
                now,
            ))
        sql = """
            INSERT OR REPLACE INTO bar_data
              (symbol, ts, open, high, low, close, volume, interval, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """
        with self._lock, self._connect() as con:
            con.executemany(sql, rows)
        return len(rows)

    def load_bars(
        self, symbol: str, interval: str, start: str, end: str
    ) -> pd.DataFrame | None:
        """Load bars for symbol between start and end (inclusive). Returns None if empty."""
        sql = """
            SELECT ts, open, high, low, close, volume
            FROM bar_data
            WHERE symbol=? AND interval=? AND ts>=? AND ts<=?
            ORDER BY ts
        """
        with self._lock, self._connect() as con:
            rows = con.execute(sql, (symbol, interval, start, end)).fetchall()
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").rename(columns=str.lower)
        return df

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def upsert_features(
        self, symbol: str, ts: str, feature_set: str, feat_dict: dict[str, float]
    ) -> None:
        now = _now_utc()
        sql = """
            INSERT OR REPLACE INTO features (symbol, ts, feature_set, features_json, computed_at)
            VALUES (?,?,?,?,?)
        """
        with self._lock, self._connect() as con:
            con.execute(sql, (symbol, ts, feature_set, json.dumps(feat_dict), now))

    def load_features(
        self, symbol: str, feature_set: str, start: str, end: str
    ) -> pd.DataFrame | None:
        sql = """
            SELECT ts, features_json FROM features
            WHERE symbol=? AND feature_set=? AND ts>=? AND ts<=?
            ORDER BY ts
        """
        with self._lock, self._connect() as con:
            rows = con.execute(sql, (symbol, feature_set, start, end)).fetchall()
        if not rows:
            return None
        records = []
        for r in rows:
            d = json.loads(r["features_json"])
            d["ts"] = r["ts"]
            records.append(d)
        df = pd.DataFrame(records).set_index("ts")
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_key: str,
        artifact_path: str,
        feature_contract: list[str],
        trained_on: list[str],
        train_start: str,
        train_end: str,
        train_samples: int,
        test_samples: int,
        train_accuracy: float,
        test_accuracy: float,
        test_f1: float,
        label_map: dict,
        feature_set_name: str = "daily_v1",
    ) -> int:
        """Register a new model version, returning its version number."""
        now = _now_utc()
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT COALESCE(MAX(version),0) as v FROM model_registry WHERE model_key=?",
                (model_key,),
            ).fetchone()
            next_version = int(row["v"]) + 1

            con.execute(
                """
                INSERT INTO model_registry
                  (model_key, version, artifact_path, feature_contract, feature_set_name,
                   trained_on, train_start, train_end, train_samples, test_samples,
                   train_accuracy, test_accuracy, test_f1, label_map, trained_at, is_active)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
                """,
                (
                    model_key,
                    next_version,
                    artifact_path,
                    json.dumps(feature_contract),
                    feature_set_name,
                    json.dumps(trained_on),
                    train_start,
                    train_end,
                    train_samples,
                    test_samples,
                    train_accuracy,
                    test_accuracy,
                    test_f1,
                    json.dumps(label_map),
                    now,
                ),
            )
        return next_version

    def deactivate_old_models(self, model_key: str, keep_version: int) -> None:
        """Mark all versions of model_key except keep_version as inactive."""
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE model_registry SET is_active=0 WHERE model_key=? AND version!=?",
                (model_key, keep_version),
            )

    def get_active_model(self, model_key: str) -> dict | None:
        """Return metadata dict for the active version of model_key, or None."""
        sql = """
            SELECT * FROM model_registry
            WHERE model_key=? AND is_active=1
            ORDER BY version DESC LIMIT 1
        """
        with self._lock, self._connect() as con:
            row = con.execute(sql, (model_key,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["feature_contract"] = json.loads(d["feature_contract"])
        d["trained_on"] = json.loads(d["trained_on"])
        d["label_map"] = json.loads(d["label_map"])
        return d

    def list_models(self, model_key: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM model_registry"
        params: tuple = ()
        if model_key:
            sql += " WHERE model_key=?"
            params = (model_key,)
        sql += " ORDER BY model_key, version"
        with self._lock, self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([dict(r) for r in rows])

    # ------------------------------------------------------------------
    # Backtest runs
    # ------------------------------------------------------------------

    def insert_backtest_run(
        self,
        run_id: str,
        symbol: str,
        strategy: str,
        data_start: str,
        data_end: str,
        start_cash: float,
        model_version: int | None = None,
        final_equity: float | None = None,
        total_return_pct: float | None = None,
        daily_sharpe: float | None = None,
        daily_sortino: float | None = None,
        max_drawdown_pct: float | None = None,
        n_round_trades: int | None = None,
        hit_rate: float | None = None,
        profit_factor: float | None = None,
        commission_per_share: float | None = None,
        slippage_bps: float | None = None,
        stop_loss_pct: float | None = None,
        holding_period: int | None = None,
    ) -> int:
        """Insert a backtest run record, returning its row id."""
        now = _now_utc()
        with self._lock, self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO backtest_runs
                  (run_id, symbol, strategy, model_version, data_start, data_end,
                   start_cash, final_equity, total_return_pct, daily_sharpe, daily_sortino,
                   max_drawdown_pct, n_round_trades, hit_rate, profit_factor,
                   commission_per_share, slippage_bps, stop_loss_pct, holding_period, ran_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id, symbol, strategy, model_version, data_start, data_end,
                    start_cash, final_equity, total_return_pct, daily_sharpe, daily_sortino,
                    max_drawdown_pct, n_round_trades, hit_rate, profit_factor,
                    commission_per_share, slippage_bps, stop_loss_pct, holding_period, now,
                ),
            )
            return cur.lastrowid

    def insert_trades(self, backtest_run_id: int, trades: pd.DataFrame) -> None:
        """Bulk-insert trade_log rows from a DataFrame with standard columns."""
        if trades.empty:
            return
        rows = []
        for _, t in trades.iterrows():
            ts_val = t.get("ts", t.get("timestamp", ""))
            ts_str = ts_val.isoformat() if hasattr(ts_val, "isoformat") else str(ts_val)
            rows.append((
                backtest_run_id,
                str(t.get("symbol", "")),
                str(t.get("strategy", "")),
                ts_str,
                str(t.get("side", "")),
                int(t.get("shares", 0)),
                float(t.get("fill_price", 0)),
                float(t.get("commission", 0)),
                float(t.get("spread_cost", 0)),
                t.get("exit_reason"),
            ))
        sql = """
            INSERT INTO trade_log
              (backtest_run_id, symbol, strategy, ts, side, shares, fill_price,
               commission, spread_cost, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        with self._lock, self._connect() as con:
            con.executemany(sql, rows)

    def get_backtest_runs(
        self,
        symbol: str | None = None,
        strategy: str | None = None,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        conditions, params = [], []
        if symbol:
            conditions.append("symbol=?"); params.append(symbol)
        if strategy:
            conditions.append("strategy=?"); params.append(strategy)
        if run_id:
            conditions.append("run_id=?"); params.append(run_id)
        sql = "SELECT * FROM backtest_runs"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY ran_at DESC"
        with self._lock, self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Daily predictions
    # ------------------------------------------------------------------

    def upsert_prediction(
        self,
        symbol: str,
        model_key: str,
        model_version: int,
        prediction_date: str,
        signal: str,
        confidence: float,
        price: float | None = None,
    ) -> None:
        now = _now_utc()
        sql = """
            INSERT INTO daily_predictions
              (symbol, model_key, model_version, prediction_date, signal,
               confidence, price_at_prediction, predicted_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(symbol, model_key, prediction_date)
            DO UPDATE SET
              signal=excluded.signal,
              confidence=excluded.confidence,
              price_at_prediction=excluded.price_at_prediction,
              model_version=excluded.model_version,
              predicted_at=excluded.predicted_at
        """
        with self._lock, self._connect() as con:
            con.execute(sql, (symbol, model_key, model_version, prediction_date,
                               signal, confidence, price, now))

    def get_predictions(self, prediction_date: str) -> pd.DataFrame:
        sql = """
            SELECT * FROM daily_predictions WHERE prediction_date=? ORDER BY symbol, model_key
        """
        with self._lock, self._connect() as con:
            rows = con.execute(sql, (prediction_date,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def new_run_id() -> str:
        return str(uuid.uuid4())
