#!/usr/bin/env python3
"""
trade_tracker.py

Tracks trading recommendations and actual execution for the current session.
Stores only current session data (daily reset), comparing recommended vs actual trades.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class TradeTracker:
    """Session-based trade tracking (no permanent history)."""
    
    def __init__(self, db_path: str = "results/trade_tracker.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize session-based database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Table for recommendations
        c.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY,
                message_id INTEGER UNIQUE,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL,
                strategies TEXT,
                posted_at TIMESTAMP,
                confirmed INTEGER DEFAULT 0,
                confirmed_by INTEGER,
                confirmed_at TIMESTAMP,
                rejected INTEGER DEFAULT 0,
                rejected_by INTEGER,
                rejected_at TIMESTAMP,
                executed INTEGER DEFAULT 0,
                session_date TEXT
            )
        ''')
        
        # Table for actual trades
        c.execute('''
            CREATE TABLE IF NOT EXISTS actual_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL,
                shares INTEGER,
                strategy TEXT,
                user_id INTEGER,
                executed_at TIMESTAMP,
                matched_recommendation_id INTEGER,
                pnl REAL,
                session_date TEXT,
                FOREIGN KEY(matched_recommendation_id) REFERENCES recommendations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_recommendation(self, symbol: str, signal: str, confidence: float, 
                         strategies: Dict, message_id: int, timestamp: datetime):
        """Add a new recommendation."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR IGNORE INTO recommendations 
            (message_id, symbol, signal, confidence, strategies, posted_at, session_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            message_id, symbol, signal, confidence,
            json.dumps(strategies), timestamp, datetime.now().date()
        ))
        
        conn.commit()
        conn.close()
    
    def confirm_recommendation(self, message_id: int, user_id: int):
        """Mark recommendation as confirmed by user."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE recommendations
            SET confirmed = 1, confirmed_by = ?, confirmed_at = ?
            WHERE message_id = ?
        ''', (user_id, datetime.now(), message_id))
        
        conn.commit()
        conn.close()
    
    def reject_recommendation(self, message_id: int, user_id: int):
        """Mark recommendation as rejected by user."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE recommendations
            SET rejected = 1, rejected_by = ?, rejected_at = ?
            WHERE message_id = ?
        ''', (user_id, datetime.now(), message_id))
        
        conn.commit()
        conn.close()
    
    def log_actual_trade(self, symbol: str, side: str, price: float, shares: int, 
                        strategy: str, user_id: int) -> int:
        """Log an actual trade execution."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Try to match with a confirmed recommendation
        c.execute('''
            SELECT id FROM recommendations
            WHERE symbol = ? AND signal = ? AND confirmed = 1 AND executed = 0
            ORDER BY confirmed_at DESC LIMIT 1
        ''', (symbol, "BUY" if side == "BUY" else "SELL"))
        
        matched_rec = c.fetchone()
        matched_id = matched_rec[0] if matched_rec else None
        
        # Insert trade
        c.execute('''
            INSERT INTO actual_trades
            (symbol, side, price, shares, strategy, user_id, executed_at, 
             matched_recommendation_id, session_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol, side, price, shares, strategy, user_id, datetime.now(),
            matched_id, datetime.now().date()
        ))
        
        trade_id = c.lastrowid
        
        # Mark recommendation as executed
        if matched_id:
            c.execute('''
                UPDATE recommendations SET executed = 1 WHERE id = ?
            ''', (matched_id,))
        
        conn.commit()
        conn.close()
        
        return trade_id
    
    def get_session_stats(self) -> Dict:
        """Get statistics for current session."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        session_date = datetime.now().date()
        
        # Count recommendations
        c.execute('SELECT COUNT(*) FROM recommendations WHERE session_date = ?', (session_date,))
        total_recs = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM recommendations WHERE session_date = ? AND confirmed = 1', (session_date,))
        confirmed_recs = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM recommendations WHERE session_date = ? AND rejected = 1', (session_date,))
        rejected_recs = c.fetchone()[0]
        
        # Count actual trades
        c.execute('SELECT COUNT(*) FROM actual_trades WHERE session_date = ?', (session_date,))
        total_trades = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM actual_trades WHERE session_date = ? AND side = "BUY"', (session_date,))
        buy_count = c.fetchone()[0]
        
        c.execute('SELECT COUNT(*) FROM actual_trades WHERE session_date = ? AND side = "SELL"', (session_date,))
        sell_count = c.fetchone()[0]
        
        # Count executed recommendations
        c.execute('SELECT COUNT(*) FROM recommendations WHERE session_date = ? AND executed = 1', (session_date,))
        executed_recs = c.fetchone()[0]
        
        # Calculate execution rate
        execution_rate = executed_recs / confirmed_recs if confirmed_recs > 0 else 0
        
        # Calculate P&L (simple: buy-exit pair)
        c.execute('''
            SELECT SUM((SELECT AVG(price) FROM actual_trades WHERE symbol = r.symbol AND side = "SELL" AND executed_at > r.confirmed_at) 
                       - (SELECT AVG(price) FROM actual_trades WHERE symbol = r.symbol AND side = "BUY" AND executed_at <= r.confirmed_at)) * 
                   (SELECT SUM(shares) FROM actual_trades WHERE symbol = r.symbol AND side = "BUY")
            FROM recommendations r
            WHERE r.session_date = ? AND r.confirmed = 1
        ''', (session_date,))
        
        pnl_result = c.fetchone()
        total_pnl = float(pnl_result[0] or 0)
        
        # Win rate (trades with positive return)
        c.execute('''
            SELECT COUNT(*) FROM (
                SELECT symbol, 
                       (SELECT AVG(price) FROM actual_trades WHERE symbol = outer.symbol AND side = "SELL") -
                       (SELECT AVG(price) FROM actual_trades WHERE symbol = outer.symbol AND side = "BUY") as trade_return
                FROM (SELECT DISTINCT symbol FROM actual_trades WHERE session_date = ?) outer
            )
            WHERE trade_return > 0
        ''', (session_date,))
        
        winning_trades = c.fetchone()[0]
        win_rate = winning_trades / total_trades if total_trades > 0 else None
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        conn.close()
        
        return {
            "total_recommendations": total_recs,
            "confirmed_recommendations": confirmed_recs,
            "rejected_recommendations": rejected_recs,
            "total_actual_trades": total_trades,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "executed_recommendations": executed_recs,
            "execution_rate": execution_rate,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "session_date": str(session_date)
        }
    
    def clear_session(self):
        """Clear all session data (for daily reset)."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        session_date = datetime.now().date()
        
        c.execute('DELETE FROM actual_trades WHERE session_date = ?', (session_date,))
        c.execute('DELETE FROM recommendations WHERE session_date = ?', (session_date,))
        
        conn.commit()
        conn.close()
    
    def get_recommendation_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recommendation history."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if symbol:
            c.execute('''
                SELECT id, symbol, signal, confidence, confirmed, executed, posted_at
                FROM recommendations
                WHERE symbol = ?
                ORDER BY posted_at DESC LIMIT ?
            ''', (symbol, limit))
        else:
            c.execute('''
                SELECT id, symbol, signal, confidence, confirmed, executed, posted_at
                FROM recommendations
                ORDER BY posted_at DESC LIMIT ?
            ''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "symbol": row[1],
                "signal": row[2],
                "confidence": row[3],
                "confirmed": bool(row[4]),
                "executed": bool(row[5]),
                "timestamp": row[6]
            }
            for row in rows
        ]
