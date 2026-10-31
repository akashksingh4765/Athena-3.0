from __future__ import annotations

import os

import json

import uuid

import io

import inspect

import logging

import contextlib

import time

import threading

import webbrowser

from datetime import datetime

from urllib.parse import urlparse, parse_qs

import keyring

import streamlit as st

from selenium import webdriver

from selenium.webdriver.chrome.options import Options

from selenium.common.exceptions import (
    NoSuchWindowException,
    WebDriverException,
)

from NorenRestApiPy.NorenApi import NorenApi

from model_manager_v1 import load_all_models_cached

import sys

from pathlib import Path

# ============================================================

# ATHENA REPORTING LAYER (embedded — no separate folder needed)

# The append-only ledger is the source of truth; the daily report

# is always regenerated from it.

# ============================================================

import math

from collections import Counter, defaultdict

from datetime import date

# ---------------- reporting: instrument_profiles.py ----------------

# Instrument profiles: leverage rules, lot sizes and charge rates.
#
# Leverage is NOT a fixed 5x. It depends on what is actually traded:
#
# * NIFTY 50 stocks, intraday (MIS): broker leverage up to 5x
#   (20% margin) at most; stock-specific and can be lower.
# * NIFTY index futures: no extra broker leverage. Full exchange
#   margin (SPAN + exposure) must be posted. Notional leverage is
#   100 / margin%, and positions are whole lots only.
# * NIFTY index options, buying: premium is paid in full (1x on capital).
#
# All rates below are DEFAULTS. Verify them against Shoonya's current
# margin calculator and charges page and edit here if they differ.
# Lot size: NIFTY = 65 from the Jan-2026 series (NSE reviews lot sizes
# roughly every six months, so re-check periodically).
# STT on F&O: futures 0.05% (sell), options 0.15% of premium (sell),
# effective 1 April 2026.



# Percent values are in PERCENT (0.05 means 0.05%).

INSTRUMENTS = {
    "NIFTY_FUTURES": {
        "label": "NIFTY index futures",
        "leverage_mode": "exchange_margin",
        "default_margin_pct": 12.0,
        "min_margin_pct": 5.0,
        "max_margin_pct": 30.0,
        "lot_size": 65,
        "note": (
            "No broker leverage: the full exchange margin (SPAN + exposure) "
            "is required. Notional leverage = 100 / margin %. Whole lots "
            "only. Margin % is an approximate default; check Shoonya's "
            "margin calculator."
        ),
        "charges": {
            "brokerage_per_order": 0.0,
            "stt_sell_pct": 0.05,
            "txn_pct": 0.00173,
            "sebi_pct": 0.0001,
            "stamp_buy_pct": 0.002,
            "gst_pct": 18.0,
        },
    },
    "NIFTY50_STOCKS_MIS": {
        "label": "NIFTY 50 stocks — intraday (MIS)",
        "leverage_mode": "broker",
        "max_leverage": 5.0,
        "default_leverage": 5.0,
        "lot_size": 1,
        "note": (
            "Up to 5x (20% margin) is the ceiling for intraday equity. "
            "Actual leverage is stock-specific and can be lower for "
            "volatile stocks. The NIFTY index itself cannot be bought; "
            "this profile is for its constituent stocks."
        ),
        "charges": {
            "brokerage_per_order": 0.0,
            "stt_sell_pct": 0.025,
            "txn_pct": 0.00297,
            "sebi_pct": 0.0001,
            "stamp_buy_pct": 0.003,
            "gst_pct": 18.0,
        },
    },
    "NIFTY_OPTIONS_BUY": {
        "label": "NIFTY index options — buying",
        "leverage_mode": "none",
        "max_leverage": 1.0,
        "default_leverage": 1.0,
        "lot_size": 65,
        "note": (
            "Premium is paid in full: 1x on capital. Prices used for "
            "sizing must be OPTION PREMIUMS, not the index level."
        ),
        "charges": {
            "brokerage_per_order": 0.0,
            "stt_sell_pct": 0.15,
            "txn_pct": 0.03503,
            "sebi_pct": 0.0001,
            "stamp_buy_pct": 0.003,
            "gst_pct": 18.0,
        },
    },
}

DEFAULT_INSTRUMENT = "NIFTY_FUTURES"


def instrument_label(key: str) -> str:
    return INSTRUMENTS[key]["label"]


def leverage_bounds(key: str) -> tuple[float, float]:
    """(min, max) user-selectable leverage for broker-leverage profiles."""
    profile = INSTRUMENTS[key]
    if profile["leverage_mode"] == "broker":
        return 1.0, float(profile["max_leverage"])
    return 1.0, 1.0


def effective_max_leverage(key: str, leverage: float, margin_pct: float) -> float:
    """Maximum notional leverage the profile allows with current settings."""
    profile = INSTRUMENTS[key]
    mode = profile["leverage_mode"]
    if mode == "broker":
        low, high = leverage_bounds(key)
        return max(low, min(float(leverage), high))
    if mode == "exchange_margin":
        return 100.0 / max(float(margin_pct), 0.01)
    return 1.0


def max_exposure(key: str, capital: float, leverage: float, margin_pct: float) -> float:
    """Theoretical maximum notional exposure (before whole-lot rounding)."""
    return float(capital) * effective_max_leverage(key, leverage, margin_pct)


def size_position(
    key: str,
    capital: float,
    price: float,
    leverage: float,
    margin_pct: float,
    lot_size: int | None = None,
) -> dict:
    """
    Size a new position. Returns quantity, exposure, margin used and
    effective leverage, or quantity 0 with a reason.
    """
    profile = INSTRUMENTS[key]
    mode = profile["leverage_mode"]
    lot = int(lot_size or profile["lot_size"] or 1)
    result = {
        "quantity": 0,
        "lots": 0,
        "exposure": 0.0,
        "margin_used": 0.0,
        "leverage": 0.0,
        "reason": "",
    }

    if price <= 0:
        result["reason"] = "No valid price."
        return result
    if capital <= 0:
        result["reason"] = "No capital available."
        return result

    if mode == "broker":
        lev = effective_max_leverage(key, leverage, margin_pct)
        quantity = int(math.floor(capital * lev / price))
        if quantity <= 0:
            result["reason"] = (
                f"Capital ₹{capital:,.2f} at {lev:.2f}x cannot buy one share "
                f"at ₹{price:,.2f}."
            )
            return result
        exposure = quantity * price
        margin = exposure / lev
        result.update(
            quantity=quantity,
            lots=quantity,
            exposure=exposure,
            margin_used=margin,
            leverage=exposure / capital,
        )
        return result

    if mode == "exchange_margin":
        margin_per_lot = price * lot * float(margin_pct) / 100.0
        lots = int(math.floor(capital / margin_per_lot)) if margin_per_lot > 0 else 0
        if lots <= 0:
            result["reason"] = (
                f"Insufficient capital for one lot: needs about "
                f"₹{margin_per_lot:,.0f} margin "
                f"({lot} × {price:,.2f} × {margin_pct:.1f}%), "
                f"available ₹{capital:,.2f}."
            )
            return result
        quantity = lots * lot
        exposure = quantity * price
        result.update(
            quantity=quantity,
            lots=lots,
            exposure=exposure,
            margin_used=lots * margin_per_lot,
            leverage=exposure / capital,
        )
        return result

    # Options buying: premium paid in full.
    cost_per_lot = price * lot
    lots = int(math.floor(capital / cost_per_lot)) if cost_per_lot > 0 else 0
    if lots <= 0:
        result["reason"] = (
            f"Insufficient capital for one lot: premium cost "
            f"₹{cost_per_lot:,.2f}, available ₹{capital:,.2f}."
        )
        return result
    quantity = lots * lot
    exposure = quantity * price
    result.update(
        quantity=quantity,
        lots=lots,
        exposure=exposure,
        margin_used=exposure,
        leverage=exposure / capital,
    )
    return result


def compute_charges(key: str, buy_value: float, sell_value: float, orders: int = 2) -> dict:
    """
    Estimate statutory charges for one round trip.
    buy_value / sell_value are turnover values in rupees.
    """
    rates = INSTRUMENTS[key]["charges"]
    turnover = float(buy_value) + float(sell_value)

    brokerage = rates["brokerage_per_order"] * orders
    stt = float(sell_value) * rates["stt_sell_pct"] / 100.0
    exchange_txn = turnover * rates["txn_pct"] / 100.0
    sebi = turnover * rates["sebi_pct"] / 100.0
    stamp = float(buy_value) * rates["stamp_buy_pct"] / 100.0
    gst = (brokerage + exchange_txn + sebi) * rates["gst_pct"] / 100.0

    taxes_fees = stt + exchange_txn + sebi + stamp + gst
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_txn": round(exchange_txn, 2),
        "sebi": round(sebi, 2),
        "stamp": round(stamp, 2),
        "gst": round(gst, 2),
        "taxes_fees": round(taxes_fees, 2),
        "total": round(brokerage + taxes_fees, 2),
    }


# ---------------- reporting: trade_ledger.py ----------------

# Append-only event/trade ledger (JSON Lines, one file per day per mode).
#
# This file is the SOURCE OF TRUTH. Events are only ever appended;
# reports are regenerated from it at any time.
#
# Event types used by Athena:
#     ENGINE_START, ENGINE_STOP, KILL_SWITCH, KILL_RESET,
#     SIGNAL, TRADE_OPEN, TRADE_CLOSE, TRADE_REJECTED,
#     EXPOSURE, RISK_VIOLATION, SETTINGS




class TradeLedger:

    def __init__(self, base_dir: str | os.PathLike):
        self.base_dir = Path(base_dir)
        self._lock = threading.Lock()

    def _path(self, mode: str, day: date) -> Path:
        return self.base_dir / mode.upper() / f"{day.isoformat()}.jsonl"

    def append(self, mode: str, event_type: str, payload: dict | None = None) -> dict:
        now = datetime.now()
        event = {
            "event_id": uuid.uuid4().hex,
            "ts": now.isoformat(timespec="seconds"),
            "mode": mode.upper(),
            "type": event_type,
            "data": payload or {},
        }
        path = self._path(mode, now.date())
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    def read_day(self, mode: str, day: date) -> list[dict]:
        path = self._path(mode, day)
        if not path.exists():
            return []
        events = []
        with self._lock:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A partially written last line must not break reports.
                        continue
        events.sort(key=lambda e: e.get("ts", ""))
        return events

    def available_days(self, mode: str) -> list[date]:
        folder = self.base_dir / mode.upper()
        if not folder.exists():
            return []
        days = []
        for file in folder.glob("*.jsonl"):
            try:
                days.append(date.fromisoformat(file.stem))
            except ValueError:
                continue
        return sorted(days, reverse=True)


# ---------------- reporting: exposure_tracker.py ----------------

# Exposure timeline: capital deployment through the day, built from
# TRADE_OPEN / TRADE_CLOSE / EXPOSURE ledger events.




def _ts(event: dict) -> datetime:
    return datetime.fromisoformat(event["ts"])


def build_exposure_timeline(events: list[dict]) -> list[dict]:
    """One row per state change / snapshot: time, state, exposure, leverage."""
    rows = []
    for event in events:
        etype = event["type"]
        data = event.get("data", {})
        if etype == "ENGINE_START":
            rows.append({
                "time": _ts(event), "state": "FLAT", "direction": "",
                "exposure": 0.0, "leverage": 0.0, "margin_used": 0.0,
                "note": "Engine started",
            })
        elif etype == "TRADE_OPEN":
            count = int(data.get("open_positions", 1))
            rows.append({
                "time": _ts(event),
                "state": f"{count} OPEN",
                "direction": data.get("direction", ""),
                "exposure": float(data.get("portfolio_exposure", data.get("exposure", 0.0))),
                "leverage": float(data.get("portfolio_leverage", data.get("leverage", 0.0))),
                "margin_used": float(data.get("portfolio_margin", data.get("margin_used", 0.0))),
                "note": f"Entry {data.get('direction', '')} {data.get('quantity', '')}",
            })
        elif etype == "EXPOSURE":
            direction = data.get("direction", "NONE")
            count = int(data.get("open_positions", 0 if direction in ("", "NONE") else 1))
            rows.append({
                "time": _ts(event),
                "state": "FLAT" if count == 0 else f"{count} OPEN",
                "direction": "" if direction in ("", "NONE") else direction,
                "exposure": float(data.get("exposure", 0.0)),
                "leverage": float(data.get("leverage", 0.0)),
                "margin_used": float(data.get("margin_used", 0.0)),
                "note": "Snapshot",
            })
        elif etype == "TRADE_CLOSE":
            count = int(data.get("open_positions", 0))
            rows.append({
                "time": _ts(event),
                "state": "FLAT" if count == 0 else f"{count} OPEN",
                "direction": "",
                "exposure": float(data.get("portfolio_exposure", 0.0)),
                "leverage": float(data.get("portfolio_leverage", 0.0)),
                "margin_used": float(data.get("portfolio_margin", 0.0)),
                "note": f"Exit ({data.get('exit_reason', '')})",
            })
        elif etype in ("ENGINE_STOP", "KILL_SWITCH"):
            last = rows[-1] if rows else {"exposure": 0.0, "leverage": 0.0, "margin_used": 0.0}
            rows.append({
                "time": _ts(event), "state": "STOPPED", "direction": "",
                "exposure": last["exposure"], "leverage": last["leverage"],
                "margin_used": last["margin_used"],
                "note": "Engine stopped" if etype == "ENGINE_STOP" else "Kill switch",
            })
    return rows


def exposure_summary(timeline: list[dict], capital: float) -> dict:
    """Peak/average exposure and leverage, utilisation, longest flat period."""
    summary = {
        "peak_exposure": 0.0,
        "avg_exposure": 0.0,
        "peak_leverage": 0.0,
        "avg_leverage_in_position": 0.0,
        "peak_margin_used": 0.0,
        "max_capital_utilization_pct": 0.0,
        "longest_flat_minutes": 0.0,
    }
    if not timeline:
        return summary

    summary["peak_exposure"] = max(r["exposure"] for r in timeline)
    summary["peak_leverage"] = max(r["leverage"] for r in timeline)
    summary["peak_margin_used"] = max(r["margin_used"] for r in timeline)
    if capital > 0:
        summary["max_capital_utilization_pct"] = (
            summary["peak_margin_used"] / capital * 100.0
        )

    # Time-weighted averages between consecutive rows.
    total_seconds = 0.0
    exposure_seconds = 0.0
    position_seconds = 0.0
    leverage_seconds = 0.0
    flat_start = None
    longest_flat = 0.0

    for current, nxt in zip(timeline, timeline[1:]):
        seconds = max((nxt["time"] - current["time"]).total_seconds(), 0.0)
        total_seconds += seconds
        exposure_seconds += current["exposure"] * seconds
        if current["exposure"] > 0:
            position_seconds += seconds
            leverage_seconds += current["leverage"] * seconds

    for row in timeline:
        if row["state"] == "FLAT":
            if flat_start is None:
                flat_start = row["time"]
        else:
            if flat_start is not None:
                longest_flat = max(
                    longest_flat, (row["time"] - flat_start).total_seconds()
                )
                flat_start = None

    if total_seconds > 0:
        summary["avg_exposure"] = exposure_seconds / total_seconds
    if position_seconds > 0:
        summary["avg_leverage_in_position"] = leverage_seconds / position_seconds
    summary["longest_flat_minutes"] = longest_flat / 60.0
    return summary


# ---------------- reporting: performance_metrics.py ----------------

# Performance, P&L decomposition, model/ensemble and risk metrics,
# all computed from ledger events.



DIRECTIONS = ("LONG", "SHORT")
CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def _bucket_label(confidence: float) -> str:
    for low, high in CONFIDENCE_BUCKETS:
        if low <= confidence < high:
            return f"{low:.0%}–{min(high, 1.0):.0%}"
    return "unknown"


def closed_trades(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if e["type"] == "TRADE_CLOSE"]


def trade_statistics(trades: list[dict]) -> dict:
    nets = [float(t.get("net_pnl", 0.0)) for t in trades]
    winners = [n for n in nets if n > 0]
    losers = [n for n in nets if n <= 0]

    gross_profit = sum(winners)
    gross_loss = abs(sum(losers))

    longest_losing_streak = 0
    streak = 0
    for net in nets:
        if net <= 0:
            streak += 1
            longest_losing_streak = max(longest_losing_streak, streak)
        else:
            streak = 0

    holding = [float(t.get("holding_minutes", 0.0)) for t in trades]

    return {
        "trades": len(trades),
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": (len(winners) / len(trades)) if trades else 0.0,
        "avg_winner": (gross_profit / len(winners)) if winners else 0.0,
        "avg_loser": (-gross_loss / len(losers)) if losers else 0.0,
        "profit_factor": (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0.0
        ),
        "largest_winner": max(nets) if nets else 0.0,
        "largest_loser": min(nets) if nets else 0.0,
        "long_pnl": sum(float(t.get("net_pnl", 0.0)) for t in trades if t.get("direction") == "LONG"),
        "short_pnl": sum(float(t.get("net_pnl", 0.0)) for t in trades if t.get("direction") == "SHORT"),
        "max_consecutive_losses": longest_losing_streak,
        "avg_holding_minutes": (sum(holding) / len(holding)) if holding else 0.0,
    }


def pnl_decomposition(trades: list[dict]) -> dict:
    gross = sum(float(t.get("gross_pnl", 0.0)) for t in trades)
    brokerage = sum(float(t.get("brokerage", 0.0)) for t in trades)
    taxes = sum(float(t.get("taxes_fees", 0.0)) for t in trades)
    slippage = sum(float(t.get("slippage", 0.0)) for t in trades)
    penalties = sum(float(t.get("penalty", 0.0)) for t in trades)
    net = sum(float(t.get("net_pnl", 0.0)) for t in trades)
    other = round(gross - brokerage - taxes - slippage - penalties - net, 2)
    return {
        "gross_pnl": gross,
        "brokerage": brokerage,
        "taxes_fees": taxes,
        "slippage": slippage,
        "penalties": penalties,
        "other_costs": other if abs(other) >= 0.01 else 0.0,
        "net_pnl": net,
    }


def equity_and_drawdown(events: list[dict], starting_capital: float) -> dict:
    """Equity path from realised P&L plus unrealised P&L snapshots."""
    realized = 0.0
    unrealized = 0.0
    peak = starting_capital
    max_dd = 0.0
    peak_at_max_dd = starting_capital
    curve = []

    for event in events:
        etype = event["type"]
        data = event.get("data", {})
        if etype == "TRADE_CLOSE":
            realized += float(data.get("net_pnl", 0.0))
            unrealized = float(data.get("portfolio_unrealized", 0.0))
        elif etype == "EXPOSURE":
            unrealized = float(data.get("unrealized_pnl", 0.0))
        elif etype == "TRADE_OPEN":
            unrealized = float(data.get("portfolio_unrealized", unrealized))
        else:
            continue
        equity = starting_capital + realized + unrealized
        curve.append({"time": event["ts"], "equity": equity})
        peak = max(peak, equity)
        drawdown = equity - peak
        if drawdown < max_dd:
            max_dd = drawdown
            peak_at_max_dd = peak

    final_equity = starting_capital + realized + unrealized
    return {
        "equity_curve": curve,
        "max_drawdown": max_dd,
        "recovered_from_drawdown": (max_dd == 0.0) or (final_equity >= peak_at_max_dd),
        "final_equity": final_equity,
        "open_unrealized": unrealized,
    }


def model_statistics(events: list[dict], trades: list[dict], model_names: list[str]) -> dict:
    signals = [e["data"] for e in events if e["type"] == "SIGNAL"]

    per_model = {
        name: {"Signals": 0, "Long": 0, "Short": 0, "Neutral": 0, "Trades": 0}
        for name in model_names
    }
    agreement_scores = []
    directional = 0
    neutral = 0
    executed = 0
    rejected = 0
    confidence_dist = defaultdict(int)

    for sig in signals:
        votes = sig.get("model_votes", {}) or {}
        ensemble = sig.get("ensemble_signal", "NEUTRAL")
        for name, vote in votes.items():
            stats = per_model.setdefault(
                name, {"Signals": 0, "Long": 0, "Short": 0, "Neutral": 0, "Trades": 0}
            )
            stats["Signals"] += 1
            key = {"LONG": "Long", "SHORT": "Short"}.get(str(vote).upper(), "Neutral")
            stats[key] += 1
        if votes:
            agree = sum(1 for v in votes.values() if str(v).upper() == ensemble)
            agreement_scores.append(agree / len(votes))
        if ensemble in DIRECTIONS:
            directional += 1
        else:
            neutral += 1
        decision = str(sig.get("decision", "")).upper()
        if decision == "EXECUTED":
            executed += 1
        elif decision.startswith("REJECTED"):
            rejected += 1
        confidence_dist[_bucket_label(float(sig.get("confidence", 0.0)))] += 1

    for trade in trades:
        for name in trade.get("models_agreed", []) or []:
            if name in per_model:
                per_model[name]["Trades"] += 1

    rejected += sum(1 for e in events if e["type"] == "TRADE_REJECTED")

    pnl_by_confidence = defaultdict(lambda: {"trades": 0, "net_pnl": 0.0})
    pnl_by_agreement = defaultdict(lambda: {"trades": 0, "net_pnl": 0.0})
    for trade in trades:
        bucket = _bucket_label(float(trade.get("confidence", 0.0)))
        pnl_by_confidence[bucket]["trades"] += 1
        pnl_by_confidence[bucket]["net_pnl"] += float(trade.get("net_pnl", 0.0))
        total_models = len(trade.get("model_votes", {}) or {}) or len(model_names)
        agree_key = f"{int(trade.get('agreement_count', 0))}/{total_models}"
        pnl_by_agreement[agree_key]["trades"] += 1
        pnl_by_agreement[agree_key]["net_pnl"] += float(trade.get("net_pnl", 0.0))

    return {
        "per_model": per_model,
        "signals": len(signals),
        "agreement_rate": (
            sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.0
        ),
        "directional_signals": directional,
        "neutral_signals": neutral,
        "trades_generated": executed,
        "trades_rejected": rejected,
        "confidence_distribution": dict(confidence_dist),
        "pnl_by_confidence": {k: dict(v) for k, v in pnl_by_confidence.items()},
        "pnl_by_agreement": {k: dict(v) for k, v in pnl_by_agreement.items()},
    }


def risk_statistics(
    events: list[dict],
    trades: list[dict],
    exposure: dict,
    drawdown: dict,
    max_allowed_leverage: float,
) -> dict:
    max_position = max((int(t.get("quantity", 0)) for t in trades), default=0)
    opens = [e for e in events if e["type"] == "TRADE_OPEN"]
    max_position = max(
        [max_position] + [int(e["data"].get("quantity", 0)) for e in opens]
    )

    # Simultaneous positions: count opens minus closes over time.
    open_count = 0
    max_simultaneous = 0
    for event in events:
        if event["type"] == "TRADE_OPEN":
            open_count += 1
        elif event["type"] == "TRADE_CLOSE":
            open_count = max(open_count - 1, 0)
        max_simultaneous = max(max_simultaneous, open_count)

    mae = min((float(t.get("mae", 0.0)) for t in trades), default=0.0)
    mfe = max((float(t.get("mfe", 0.0)) for t in trades), default=0.0)

    violations = [e["data"] for e in events if e["type"] == "RISK_VIOLATION"]
    leverage_breach = exposure["peak_leverage"] > max_allowed_leverage + 1e-6

    return {
        "max_intraday_drawdown": drawdown["max_drawdown"],
        "max_leverage": exposure["peak_leverage"],
        "allowed_leverage": max_allowed_leverage,
        "leverage_within_limit": not leverage_breach,
        "max_capital_utilization_pct": exposure["max_capital_utilization_pct"],
        "max_position_size": max_position,
        "max_simultaneous_positions": max_simultaneous,
        "max_consecutive_losses": trade_statistics(trades)["max_consecutive_losses"],
        "longest_flat_minutes": exposure["longest_flat_minutes"],
        "largest_adverse_excursion": mae,
        "largest_favorable_excursion": mfe,
        "kill_switch_events": sum(1 for e in events if e["type"] == "KILL_SWITCH"),
        "risk_violations": len(violations) + (1 if leverage_breach else 0),
        "risk_violation_details": violations,
        "rejected_trades": sum(1 for e in events if e["type"] == "TRADE_REJECTED"),
    }


# ---------------- reporting: daily_report.py ----------------

# End-of-day report generated from the ledger.
#
# Order: capital -> exposure -> leverage -> positions -> trades ->
# P&L -> costs -> risk -> model behaviour -> conclusion.





def _rs(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.2f}"


def _signed_rs(value: float) -> str:
    return ("+" if value > 0 else "") + _rs(value)


def build_daily_report(
    ledger: TradeLedger,
    mode: str,
    day: date,
    starting_capital: float,
    model_names: list[str],
    max_allowed_leverage: float,
) -> dict:
    events = ledger.read_day(mode, day)

    starts = [e for e in events if e["type"] == "ENGINE_START"]
    if starts:
        first = starts[0]["data"]
        starting_capital = float(first.get("capital", starting_capital))
        max_allowed_leverage = float(first.get("max_leverage", max_allowed_leverage))
    instruments = sorted({
        e["data"].get("instrument_label", e["data"].get("instrument", ""))
        for e in events
        if e["type"] in ("ENGINE_START", "TRADE_OPEN") and e["data"]
    } - {""})

    trades = closed_trades(events)
    timeline = build_exposure_timeline(events)
    exposure = exposure_summary(timeline, starting_capital)
    stats = trade_statistics(trades)
    pnl = pnl_decomposition(trades)
    equity = equity_and_drawdown(events, starting_capital)
    models = model_statistics(events, trades, model_names)
    risk = risk_statistics(events, trades, exposure, equity, max_allowed_leverage)

    ending_capital = starting_capital + pnl["net_pnl"]
    open_position = any(
        e["type"] == "TRADE_OPEN" for e in events
    ) and (
        sum(1 for e in events if e["type"] == "TRADE_OPEN")
        > sum(1 for e in events if e["type"] == "TRADE_CLOSE")
    )

    capital_section = {
        "Starting capital": starting_capital,
        "Ending capital (realised)": ending_capital,
        "Open unrealised P&L": equity["open_unrealized"],
        "Peak capital deployed (margin)": exposure["peak_margin_used"],
        "Unutilised capital at peak": max(starting_capital - exposure["peak_margin_used"], 0.0),
        "Total traded exposure": sum(float(t.get("exposure", 0.0)) for t in trades),
        "Peak exposure": exposure["peak_exposure"],
        "Average exposure (time-weighted)": exposure["avg_exposure"],
        "Peak portfolio leverage (open positions)": exposure["peak_leverage"],
        "Average portfolio leverage while holding": exposure["avg_leverage_in_position"],
        "Leverage cap": max_allowed_leverage,
        "Max capital utilisation %": exposure["max_capital_utilization_pct"],
    }

    trade_rows = []
    for t in trades:
        trade_rows.append({
            "Entry": t.get("entry_time", ""),
            "Exit": t.get("exit_time", ""),
            "Symbol": t.get("symbol", ""),
            "Instrument": t.get("instrument_label", ""),
            "Direction": t.get("direction", ""),
            "Qty": t.get("quantity", 0),
            "Entry Price": t.get("entry_price", 0.0),
            "Exit Price": t.get("exit_price", 0.0),
            "Capital Allocated": t.get("margin_used", 0.0),
            "Exposure": t.get("exposure", 0.0),
            "Leverage": t.get("leverage", 0.0),
            "Gross P&L": t.get("gross_pnl", 0.0),
            "Charges": t.get("charges_total", 0.0),
            "Slippage": t.get("slippage", 0.0),
            "Net P&L": t.get("net_pnl", 0.0),
            "Holding (min)": t.get("holding_minutes", 0.0),
            "Signal": t.get("signal", ""),
            "Confidence": t.get("confidence", 0.0),
            "Models Agreed": ", ".join(t.get("models_agreed", []) or []),
            "Penalty": t.get("penalty", 0.0),
            "Exit Reason": t.get("exit_reason", ""),
        })

    timeline_rows = [
        {
            "Time": r["time"].strftime("%H:%M:%S"),
            "State": r["state"],
            "Exposure": r["exposure"],
            "Leverage": r["leverage"],
            "Margin Used": r["margin_used"],
            "Note": r["note"],
        }
        for r in timeline
    ]

    conclusion = _build_conclusion(
        events, trades, pnl, stats, equity, exposure, models, risk,
        starting_capital, open_position,
    )

    return {
        "meta": {
            "date": day.isoformat(),
            "mode": mode.upper(),
            "instruments": instruments,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "event_count": len(events),
        },
        "capital_exposure": capital_section,
        "trades": trade_rows,
        "timeline": timeline_rows,
        "pnl": pnl,
        "stats": stats,
        "models": models,
        "risk": risk,
        "conclusion": conclusion,
        "open_position_at_report": open_position,
    }


def _build_conclusion(
    events, trades, pnl, stats, equity, exposure, models, risk,
    starting_capital, open_position,
) -> dict:
    net = pnl["net_pnl"]
    if not trades:
        result = "NO TRADES"
    elif net > 0:
        result = "PROFITABLE"
    elif net < 0:
        result = "LOSS"
    else:
        result = "BREAKEVEN"

    pf = stats["profit_factor"]
    pf_text = "∞" if pf == float("inf") else f"{pf:.2f}"
    total_charges = pnl["brokerage"] + pnl["taxes_fees"]

    lines = [
        f"Net P&L: {_signed_rs(net)}",
        f"Return on starting capital: "
        f"{(net / starting_capital * 100.0) if starting_capital else 0.0:+.2f}%",
        f"Trades: {stats['trades']}",
        f"Win rate: {stats['win_rate']:.1%}",
        f"Profit factor: {pf_text}",
        f"Maximum drawdown: {_rs(equity['max_drawdown'])}",
        f"Peak portfolio leverage: {exposure['peak_leverage']:.2f}×",
        f"Average portfolio leverage while holding: {exposure['avg_leverage_in_position']:.2f}×",
        f"Total charges: {_rs(total_charges)}  |  Slippage: {_rs(pnl['slippage'])}"
        f"  |  Auto square-off penalties: {_rs(pnl.get('penalties', 0.0))}",
    ]

    assessment = []
    if risk["leverage_within_limit"]:
        assessment.append(
            f"Trading stayed within the {risk['allowed_leverage']:.2f}× leverage limit."
        )
    else:
        assessment.append(
            f"LEVERAGE LIMIT BREACHED: peak {risk['max_leverage']:.2f}× "
            f"vs allowed {risk['allowed_leverage']:.2f}×."
        )

    if trades and models["pnl_by_agreement"]:
        best = max(models["pnl_by_agreement"].items(), key=lambda kv: kv[1]["net_pnl"])
        assessment.append(
            f"Best results came when {best[0]} models agreed "
            f"({best[1]['trades']} trades, {_signed_rs(best[1]['net_pnl'])})."
        )

    rejections = [e["data"].get("reason", "") for e in events if e["type"] == "TRADE_REJECTED"]
    if rejections:
        top_reason, count = Counter(rejections).most_common(1)[0]
        assessment.append(
            f"{len(rejections)} trade(s) were rejected. Most common reason: "
            f"{top_reason} ({count}×)."
        )

    penalised = [t for t in trades if float(t.get("penalty", 0.0)) > 0]
    if penalised:
        assessment.append(
            f"WARNING: {len(penalised)} position(s) were auto-squared-off by the "
            f"broker with penalties totalling {_rs(sum(float(t['penalty']) for t in penalised))}."
        )
    window_exits = Counter(
        t.get("exit_reason", "") for t in trades
        if str(t.get("exit_reason", "")).startswith(("WINDOW_", "PRE_CLOSE"))
    )
    if window_exits:
        assessment.append(
            "End-of-day exits: "
            + ", ".join(f"{k} ×{v}" for k, v in window_exits.items())
            + "."
        )

    if risk["kill_switch_events"]:
        assessment.append(f"Emergency kill switch used {risk['kill_switch_events']} time(s).")

    if open_position:
        assessment.append(
            "WARNING: a position was still open when this report was generated."
        )

    if not trades and not rejections:
        assessment.append("No trades were attempted.")

    if total_charges + pnl["slippage"] > 0 and pnl["gross_pnl"] > 0:
        share = (total_charges + pnl["slippage"]) / pnl["gross_pnl"] * 100.0
        assessment.append(f"Costs consumed {share:.1f}% of gross profit.")

    return {"result": result, "lines": lines, "assessment": assessment}


def report_to_markdown(report: dict) -> str:
    meta = report["meta"]
    out = [
        f"# Athena 3.0 — Daily {meta['mode']} Report — {meta['date']}",
        "",
        f"Instruments: {', '.join(meta['instruments']) or '—'}  ",
        f"Generated: {meta['generated_at']}  ",
        f"Ledger events: {meta['event_count']}",
        "",
        f"## Daily Result: {report['conclusion']['result']}",
        "",
    ]
    out += [f"- {line}" for line in report["conclusion"]["lines"]]
    out += ["", "**Assessment:**", ""]
    out += [f"- {line}" for line in report["conclusion"]["assessment"]]

    out += ["", "## 1. Capital & exposure", "", "| Metric | Value |", "|---|---|"]
    for key, value in report["capital_exposure"].items():
        if "leverage" in key.lower():
            text = f"{value:.2f}×"
        elif "%" in key:
            text = f"{value:.1f}%"
        else:
            text = _rs(value)
        out.append(f"| {key} | {text} |")

    out += ["", "## 2. P&L decomposition", "", "| Item | Amount |", "|---|---|"]
    p = report["pnl"]
    out += [
        f"| Gross trading P&L | {_signed_rs(p['gross_pnl'])} |",
        f"| Brokerage | {_rs(-p['brokerage'])} |",
        f"| Taxes/fees | {_rs(-p['taxes_fees'])} |",
        f"| Slippage | {_rs(-p['slippage'])} |",
        f"| Auto square-off penalties | {_rs(-p.get('penalties', 0.0))} |",
        f"| Other costs | {_rs(-p['other_costs'])} |",
        f"| **Net P&L** | **{_signed_rs(p['net_pnl'])}** |",
    ]

    s = report["stats"]
    pf = "∞" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    out += [
        "", "## 3. Trade statistics", "",
        f"- Trades: {s['trades']} (wins {s['winning_trades']}, losses {s['losing_trades']})",
        f"- Win rate: {s['win_rate']:.1%} | Profit factor: {pf}",
        f"- Average winner: {_rs(s['avg_winner'])} | Average loser: {_rs(s['avg_loser'])}",
        f"- Largest winner: {_rs(s['largest_winner'])} | Largest loser: {_rs(s['largest_loser'])}",
        f"- Long P&L: {_signed_rs(s['long_pnl'])} | Short P&L: {_signed_rs(s['short_pnl'])}",
        f"- Average holding time: {s['avg_holding_minutes']:.1f} min",
    ]

    out += ["", "## 4. Trades", ""]
    if report["trades"]:
        cols = ["Entry", "Exit", "Direction", "Qty", "Entry Price", "Exit Price",
                "Exposure", "Leverage", "Gross P&L", "Charges", "Slippage",
                "Net P&L", "Confidence", "Exit Reason"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        for t in report["trades"]:
            out.append("| " + " | ".join(str(t.get(c, "")) for c in cols) + " |")
    else:
        out.append("No closed trades.")

    out += ["", "## 5. Exposure timeline", ""]
    if report["timeline"]:
        out.append("| Time | State | Exposure | Leverage | Note |")
        out.append("|---|---|---|---|---|")
        for r in report["timeline"]:
            out.append(
                f"| {r['Time']} | {r['State']} | {_rs(r['Exposure'])} | "
                f"{r['Leverage']:.2f}× | {r['Note']} |"
            )
    else:
        out.append("No exposure recorded.")

    m = report["models"]
    out += ["", "## 6. Model / ensemble", "",
            "| Model | Signals | Long | Short | Neutral | Trades |",
            "|---|---|---|---|---|---|"]
    for name, v in m["per_model"].items():
        out.append(f"| {name} | {v['Signals']} | {v['Long']} | {v['Short']} | "
                   f"{v['Neutral']} | {v['Trades']} |")
    out += [
        "",
        f"- Ensemble signals: {m['signals']} (directional {m['directional_signals']}, "
        f"neutral {m['neutral_signals']})",
        f"- Agreement rate: {m['agreement_rate']:.1%}",
        f"- Trades generated: {m['trades_generated']} | Rejected: {m['trades_rejected']}",
        f"- Confidence distribution: {m['confidence_distribution'] or '—'}",
        f"- P&L by confidence: {m['pnl_by_confidence'] or '—'}",
        f"- P&L by model agreement: {m['pnl_by_agreement'] or '—'}",
    ]

    r = report["risk"]
    out += [
        "", "## 7. Risk", "",
        f"- Max intraday drawdown: {_rs(r['max_intraday_drawdown'])}",
        f"- Max leverage: {r['max_leverage']:.2f}× (allowed {r['allowed_leverage']:.2f}×)",
        f"- Max capital utilisation: {r['max_capital_utilization_pct']:.1f}%",
        f"- Max position size: {r['max_position_size']}",
        f"- Max simultaneous positions: {r['max_simultaneous_positions']}",
        f"- Max consecutive losses: {r['max_consecutive_losses']}",
        f"- Longest flat period: {r['longest_flat_minutes']:.1f} min",
        f"- Largest adverse excursion: {_rs(r['largest_adverse_excursion'])}",
        f"- Largest favourable excursion: {_rs(r['largest_favorable_excursion'])}",
        f"- Kill-switch events: {r['kill_switch_events']}",
        f"- Risk-rule violations: {r['risk_violations']}",
        f"- Rejected trades: {r['rejected_trades']}",
    ]
    return "\n".join(out) + "\n"

# ============================================================

# ATHENA 3.0 — UNIFIED TRADING TERMINAL V3

# PAPER + REAL ARCHITECTURE

#

# IMPORTANT:

# - PAPER execution is simulated.

# - REAL mode is architecturally prepared but ORDER SUBMISSION

#   IS LOCKED in this version.

# - Shoonya authentication/session is available.

# - No real-money order can be sent by this file.

# ============================================================

st.set_page_config(

    page_title="Athena 3.0 - Trading Terminal",

    page_icon="📈",

    layout="wide",

)

# ============================================================

# CONSTANTS

# ============================================================

PAPER_INITIAL_CAPITAL = 10_000.0

# Paper exposure is no longer a fixed 5x (₹50,000 on ₹10,000).

# It is derived from the selected instrument profile — see

# athena_reporting/instrument_profiles.py and the Paper Settings panel.

MODEL_NAMES = ["XGBoost", "LightGBM", "CatBoost", "LSTM", "Transformer"]

PAPER_DEFAULT_SLIPPAGE_PCT = 0.01

PAPER_DEFAULT_LEVERAGE_CAP = 5.0

# Each new trade may use at most this % of the portfolio leverage budget,

# so several positions can be held at once.

PAPER_DEFAULT_TRADE_ALLOCATION_PCT = 50.0

# Session times (PC clock must be IST). NSE closes at 15:30; brokers

# auto square-off intraday positions around 15:15 WITH a penalty, so

# Athena must be flat before 15:15. Verify Shoonya's exact time/penalty.

MARKET_OPEN_TIME = "09:15"

BROKER_AUTO_SQUARE_OFF_TIME = "15:15"

MARKET_CLOSE_TIME = "15:30"

EXIT_WINDOW_MINUTES = 30          # exit window opens 14:45

FORCED_EXIT_BUFFER_MINUTES = 5    # everything closed by 15:10

EXIT_TRAIL_GIVEBACK_PCT = 30.0    # winners: exit after giving back 30% of best

EXIT_LOSS_EXTENSION_PCT = 0.20    # losers: exit if loss grows by 0.20% of exposure

BROKER_AUTO_SQUARE_OFF_PENALTY = 50.0   # ₹ per position (simulated)

BROKER_PENALTY_GST_PCT = 18.0

# Back-compat alias used in captions.

PAPER_SQUARE_OFF_TIME = BROKER_AUTO_SQUARE_OFF_TIME

PAPER_EXPOSURE_SNAPSHOT_SECONDS = 60

LEDGER_DIR = Path(__file__).resolve().parent / "athena_data" / "ledger"

BROKER_NAME = "Shoonya"

DEFAULT_SYMBOL = "NIFTY 50"

NIFTY50_FEED_TOKEN = "26000"

CLIENT_ID = os.environ.get("SHOONYA_CLIENT_ID", "")

USER_ID = os.environ.get("SHOONYA_USER_ID", "")

KEYRING_SERVICE = "Athena3_Shoonya"

# FIX: these URLs were corrupted ("http&#x73;*:*//...") which made
# ChromeDriver fail with "invalid argument". Restored to valid URLs.

OAUTH_URL = "https://api.shoonya.com/OAuthlogin/authorize/oauth"

REST_HOST = "https://api.shoonya.com/NorenWClientAPI/"

WEBSOCKET_URL = "wss://api.shoonya.com/NorenWSAPI/"

REAL_ORDER_EXECUTION_ENABLED = False

# OAuth capture tuning

OAUTH_TIMEOUT_SECONDS = 300

OAUTH_POLL_INTERVAL_SECONDS = 0.3

# ============================================================

# ============================================================

# PROCESS-SHARED STATE

# ============================================================

# Streamlit session_state is browser-session scoped. Athena uses

# process-shared state so all local browser sessions control the

# same running terminal.

@st.cache_resource

def get_shared_state():

    return {"lock": threading.RLock(), "initialized": False}

SHARED_STATE = get_shared_state()

@st.cache_resource

def get_trade_ledger() -> TradeLedger:

    return TradeLedger(LEDGER_DIR)

LEDGER = get_trade_ledger()

def sync_from_shared_state() -> None:

    """Copy shared Athena state into the current Streamlit browser session."""

    with SHARED_STATE["lock"]:

        for key, value in SHARED_STATE.items():

            if key in {"lock", "initialized"}:

                continue

            st.session_state[key] = value

def persist_to_shared_state() -> None:

    """Persist current browser-session state into the shared Athena state."""

    with SHARED_STATE["lock"]:

        shared_keys = {

            key

            for key in SHARED_STATE.keys()

            if key not in {"lock", "initialized"}

        }

        for key in shared_keys:

            if key in st.session_state:

                SHARED_STATE[key] = st.session_state[key]

# ============================================================

def initialize_state() -> None:

    defaults = {

        "model_load_results": {},

        "model_loaded_count": 0,

        "model_load_error": "",

        # Global

        "trading_mode": "PAPER",

        "system_killed": False,

        "symbol": DEFAULT_SYMBOL,

        # Paper account

        "paper_running": False,

        "paper_position": "NONE",

        "paper_entry_price": 0.0,

        "paper_current_price": 0.0,

        "paper_quantity": 0,

        "paper_realized_pnl": 0.0,

        "paper_unrealized_pnl": 0.0,

        "paper_gross_pnl": 0.0,

        "paper_charges": 0.0,

        "paper_net_pnl": 0.0,

        "paper_trade_count": 0,

        "paper_winning_trades": 0,

        "paper_losing_trades": 0,

        "paper_max_drawdown": 0.0,

        "paper_peak_equity": PAPER_INITIAL_CAPITAL,

        "paper_equity": PAPER_INITIAL_CAPITAL,

        "paper_trade_log": [],

        # Paper risk / leverage settings (instrument-based, not fixed 5x)

        "paper_capital": PAPER_INITIAL_CAPITAL,

        "paper_instrument": DEFAULT_INSTRUMENT,

        "paper_leverage": float(
            INSTRUMENTS[DEFAULT_INSTRUMENT].get("default_leverage", 1.0)
        ),

        "paper_margin_pct": float(
            INSTRUMENTS["NIFTY_FUTURES"]["default_margin_pct"]
        ),

        "paper_lot_size": int(INSTRUMENTS[DEFAULT_INSTRUMENT]["lot_size"]),

        "paper_slippage_pct": PAPER_DEFAULT_SLIPPAGE_PCT,

        "paper_slippage": 0.0,

        "paper_open_positions": [],

        "paper_leverage_cap": PAPER_DEFAULT_LEVERAGE_CAP,

        "paper_trade_allocation_pct": PAPER_DEFAULT_TRADE_ALLOCATION_PCT,

        "paper_portfolio_exposure": 0.0,

        "paper_portfolio_leverage": 0.0,

        "paper_avg_open_leverage": 0.0,

        "paper_margin_in_use": 0.0,

        "paper_penalties": 0.0,

        "paper_last_snapshot": "",

        "ledger_error": "",

        # Real account display/execution state

        "real_running": False,

        "real_position": "NONE",

        "real_entry_price": 0.0,

        "real_current_price": 0.0,

        "real_quantity": 0,

        "real_trade_log": [],

        "real_pnl": 0.0,

        # Common signal/model state

        "last_signal": "NONE",

        "signal_confidence": 0.0,

        "current_price": 0.0,

        "model_status": "NOT CONNECTED",

        "data_status": "NOT CONNECTED",

        # Shoonya

        "shoonya_status": "NOT CONNECTED",

        "shoonya_session": None,

        "shoonya_access_token": None,

        "shoonya_refresh_token": None,

        "oauth_browser_active": False,

        "oauth_code": None,

        # FIX: connection messages survive st.rerun()

        "shoonya_messages": [],

        "shoonya_diag_text": "",

        # Real-mode safety

        "real_unlock_confirmed": False,

        # UI

        "last_update": "Never",

    }

    with SHARED_STATE["lock"]:

        if not SHARED_STATE["initialized"]:

            for key, value in defaults.items():

                SHARED_STATE[key] = value

            SHARED_STATE["initialized"] = True

        else:

            # FIX: add any keys introduced after the server started,

            # otherwise persist_to_shared_state() silently drops them.

            for key, value in defaults.items():

                if key not in SHARED_STATE:

                    SHARED_STATE[key] = value

    sync_from_shared_state()

def load_models_for_dashboard():

    if st.session_state.model_load_results:

        return

    st.session_state.model_status = "LOADING"

    try:

        results = load_all_models_cached()

        loaded_count = sum(

            1 for result in results.values()

            if result.loaded

        )

        st.session_state.model_load_results = results

        st.session_state.model_loaded_count = loaded_count

        st.session_state.model_load_error = ""

        if loaded_count == 5:

            st.session_state.model_status = "READY"

        else:

            st.session_state.model_status = "ERROR"

            st.session_state.model_load_error = (

                f"Only {loaded_count}/5 models loaded."

            )

    except Exception as exc:

        st.session_state.model_load_results = {}

        st.session_state.model_loaded_count = 0

        st.session_state.model_load_error = str(exc)

        st.session_state.model_status = "ERROR"

# ============================================================

# LIVE SHOONYA MARKET FEED

# ============================================================

# FIX: Streamlit re-runs this script on every refresh. A plain module

# variable would be RESET each time, so the WebSocket thread kept writing

# ticks into an old copy the UI never saw. cache_resource keeps ONE

# feed state for the whole server process.

@st.cache_resource

def _get_feed_runtime():

    return {

        "lock": threading.Lock(),

        "state": {

            "running": False,

            "connected": False,

            "last_tick": None,

            "quote": {},

            "error": None,

            "thread": None,

            "api": None,

        },

    }

_FEED_RUNTIME = _get_feed_runtime()

FEED_LOCK = _FEED_RUNTIME["lock"]

FEED_STATE = _FEED_RUNTIME["state"]

def _feed_quote_callback(message):

    if not isinstance(message, dict):

        return

    token = str(message.get("tk", ""))

    if token != NIFTY50_FEED_TOKEN:

        return

    with FEED_LOCK:

        FEED_STATE["quote"].update(message)

        FEED_STATE["last_tick"] = datetime.now()

        FEED_STATE["connected"] = True

        FEED_STATE["error"] = None

def _feed_open_callback():

    with FEED_LOCK:

        FEED_STATE["connected"] = True

        FEED_STATE["error"] = None

        api = FEED_STATE.get("api")

    if api is not None:

        try:

            api.subscribe(f"NSE|{NIFTY50_FEED_TOKEN}")

        except Exception as exc:

            with FEED_LOCK:

                FEED_STATE["error"] = f"Subscription failed: {exc}"

def _feed_order_callback(message):

    # Order updates are deliberately ignored in this data-only stage.

    return

def _feed_worker(api):

    try:

        api.start_websocket(

            order_update_callback=_feed_order_callback,

            subscribe_callback=_feed_quote_callback,

            socket_open_callback=_feed_open_callback,

        )

    except Exception as exc:

        with FEED_LOCK:

            FEED_STATE["running"] = False

            FEED_STATE["connected"] = False

            FEED_STATE["error"] = str(exc)

def start_market_feed(api=None):

    # Prefer the authenticated API object passed directly from the

    # successful Shoonya OAuth flow. Fall back to the current browser

    # session only when start_market_feed() is called independently.

    if api is None:

        api = st.session_state.get("shoonya_session")

    if api is None:

        return

    with FEED_LOCK:

        thread = FEED_STATE.get("thread")

        if (

            FEED_STATE["running"]

            and thread is not None

            and thread.is_alive()

        ):

            return

        FEED_STATE["running"] = True

        FEED_STATE["connected"] = False

        FEED_STATE["error"] = None

        FEED_STATE["api"] = api

        thread = threading.Thread(

            target=_feed_worker,

            args=(api,),

            name="AthenaShoonyaMarketFeed",

            daemon=True,

        )

        FEED_STATE["thread"] = thread

        thread.start()

def stop_market_feed():

    with FEED_LOCK:

        api = FEED_STATE.get("api")

        FEED_STATE["running"] = False

        FEED_STATE["connected"] = False

        FEED_STATE["thread"] = None

        FEED_STATE["api"] = None

    if api is not None:

        try:

            api.close_websocket()

        except Exception:

            pass

def get_market_feed_snapshot():

    with FEED_LOCK:

        return {

            "running": FEED_STATE["running"],

            "connected": FEED_STATE["connected"],

            "last_tick": FEED_STATE["last_tick"],

            "quote": dict(FEED_STATE["quote"]),

            "error": FEED_STATE["error"],

        }

def render_live_market_feed():

    snapshot = get_market_feed_snapshot()

    quote = snapshot["quote"]

    st.subheader("Live Market Data")

    if snapshot["error"]:

        st.error(f"WebSocket error: {snapshot['error']}")

    if snapshot["connected"]:

        st.success(

            "Shoonya WebSocket connected — NIFTY 50 subscription active."

        )

    elif snapshot["running"]:

        st.info("Shoonya WebSocket starting...")

    else:

        # Shoonya session lives in the process-shared state.

        _api = st.session_state.get("shoonya_session")

        if _api is None:

            with SHARED_STATE["lock"]:

                _api = SHARED_STATE.get("shoonya_session")

        if _api is None:

            st.warning(

                "Market data feed is not running. Connect Shoonya first."

            )

        elif not snapshot["error"]:

            # Self-heal: session exists but feed is not running.

            start_market_feed(_api)

            st.info("Shoonya session found — starting market feed...")

        else:

            st.warning("Shoonya is connected but the market feed stopped.")

            if st.button("🔄 RETRY MARKET FEED", key="retry_market_feed"):

                with FEED_LOCK:

                    FEED_STATE["error"] = None

                start_market_feed(_api)

                st.rerun()

    cols = st.columns(4)

    ltp = quote.get("lp", "—")

    pct = quote.get("pc", "—")

    open_price = quote.get("o", "—")

    prev_close = quote.get("c", "—")

    with cols[0]:

        st.metric("NIFTY 50 LTP", ltp)

    with cols[1]:

        st.metric("% Change", pct)

    with cols[2]:

        st.metric("Open", open_price)

    with cols[3]:

        st.metric("Previous Close", prev_close)

    last_tick = snapshot["last_tick"]

    if last_tick is None:

        st.caption(

            "Waiting for the first NIFTY 50 tick. (Ticks arrive only while "

            "NSE is open, 09:15–15:30 IST on trading days.)"

        )

    elif (datetime.now() - last_tick).total_seconds() > 60:

        st.caption(

            f"⚠️ No new tick for {int((datetime.now() - last_tick).total_seconds())}s "

            "— normal outside market hours; otherwise check the connection."

        )

    else:

        st.caption(

            "Last tick received: "

            + last_tick.strftime("%Y-%m-%d %H:%M:%S")

        )

if hasattr(st, "fragment"):

    @st.fragment(run_every="2s")

    def _live_market_feed_fragment():

        render_live_market_feed()

    _live_market_feed_fragment()

else:

    render_live_market_feed()

initialize_state()

load_models_for_dashboard()

# ============================================================

# HELPERS

# ============================================================

def format_rupees(value: float) -> str:

    return f"₹{value:,.2f}"

def status_badge(label: str, state: str) -> None:

    if state in {"OK", "CONNECTED", "READY"}:

        st.success(f"{label}: {state}")

    elif state in {

        "RUNNING",

        "PAPER",

        "AUTHENTICATING",

        "WAITING FOR OTP",

        "SESSION READY",

        "LOCKED",

    }:

        st.info(f"{label}: {state}")

    elif state in {

        "STOPPED",

        "NOT CONNECTED",

        "MARKET CLOSED",

    }:

        st.warning(f"{label}: {state}")

    else:

        st.error(f"{label}: {state}")

def update_timestamp() -> None:

    st.session_state.last_update = datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )

def active_mode() -> str:

    return st.session_state.trading_mode

def active_position() -> str:

    if active_mode() == "PAPER":

        return st.session_state.paper_position

    return st.session_state.real_position

def active_current_price() -> float:

    if active_mode() == "PAPER":

        return st.session_state.paper_current_price

    return st.session_state.real_current_price

def reset_paper_account() -> None:

    st.session_state.paper_running = False

    st.session_state.paper_position = "NONE"

    st.session_state.paper_entry_price = 0.0

    st.session_state.paper_current_price = 0.0

    st.session_state.paper_quantity = 0

    st.session_state.paper_realized_pnl = 0.0

    st.session_state.paper_unrealized_pnl = 0.0

    st.session_state.paper_gross_pnl = 0.0

    st.session_state.paper_charges = 0.0

    st.session_state.paper_net_pnl = 0.0

    st.session_state.paper_trade_count = 0

    st.session_state.paper_winning_trades = 0

    st.session_state.paper_losing_trades = 0

    st.session_state.paper_max_drawdown = 0.0

    st.session_state.paper_peak_equity = float(st.session_state.paper_capital)

    st.session_state.paper_equity = float(st.session_state.paper_capital)

    st.session_state.paper_trade_log = []

    st.session_state.paper_slippage = 0.0

    st.session_state.paper_open_positions = []

    st.session_state.paper_portfolio_exposure = 0.0

    st.session_state.paper_portfolio_leverage = 0.0

    st.session_state.paper_avg_open_leverage = 0.0

    st.session_state.paper_margin_in_use = 0.0

    st.session_state.paper_penalties = 0.0

    st.session_state.paper_last_snapshot = ""

    update_timestamp()

# ============================================================

# PAPER RISK / LEVERAGE / EXECUTION SIMULATOR + LEDGER

# ============================================================

# Multi-position paper portfolio.

#

# LEVERAGE = total exposure of positions CURRENTLY OPEN / capital.

# Closed positions are removed immediately, so they never count in the

# current leverage or in the average leverage of open positions.

# New trades are sized down (or rejected) so the portfolio never

# exceeds the leverage cap you set.

#

# SESSION CLOSE LOGIC (PC clock — must be set to IST):

#   09:15  market opens — new entries allowed

#   14:45  exit window opens (30 min before 15:15): no new entries;

#          each position exits at its best moment:

#            * losing position recovers to breakeven after costs -> exit

#            * profitable position gives back 30% of its best -> exit

#            * losing position worsens by 0.20% of exposure    -> exit

#   15:10  hard exit: everything still open is closed (5-min buffer)

#   15:15  broker auto square-off: anything STILL open is closed by the

#          broker WITH a penalty (simulated)

#   15:30  NSE market close

# Positions left over from a previous day are also force-closed with

# the penalty (intraday positions cannot be carried overnight).

from datetime import timedelta

def paper_settings() -> dict:

    key = st.session_state.paper_instrument

    ceiling = effective_max_leverage(

        key,

        st.session_state.paper_leverage,

        st.session_state.paper_margin_pct,

    )

    cap = float(st.session_state.paper_leverage_cap)

    return {

        "instrument": key,

        "instrument_label": instrument_label(key),

        "capital": float(st.session_state.paper_capital),

        "leverage": float(st.session_state.paper_leverage),

        "margin_pct": float(st.session_state.paper_margin_pct),

        "lot_size": int(st.session_state.paper_lot_size),

        "slippage_pct": float(st.session_state.paper_slippage_pct),

        "leverage_cap": cap,

        "trade_allocation_pct": float(st.session_state.paper_trade_allocation_pct),

        "instrument_ceiling": ceiling,

        # The limit actually enforced on the whole open portfolio.

        "max_leverage": min(cap, ceiling),

    }

def paper_available_capital() -> float:

    return float(st.session_state.paper_capital) + float(

        st.session_state.paper_net_pnl

    )

def ledger_event(event_type: str, payload: dict | None = None, mode: str = "PAPER"):

    try:

        return LEDGER.append(mode, event_type, payload or {})

    except Exception as exc:

        st.session_state.ledger_error = f"Ledger write failed: {exc}"

        return None

# ---------------- session schedule ----------------

def _clock(text: str, now: datetime) -> datetime:

    hour, minute = (int(x) for x in text.split(":"))

    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)

def session_schedule(now: datetime | None = None) -> dict:

    now = now or datetime.now()

    square_off = _clock(BROKER_AUTO_SQUARE_OFF_TIME, now)

    return {

        "market_open": _clock(MARKET_OPEN_TIME, now),

        "exit_window": square_off - timedelta(minutes=EXIT_WINDOW_MINUTES),

        "forced_exit": square_off - timedelta(minutes=FORCED_EXIT_BUFFER_MINUTES),

        "broker_square_off": square_off,

        "market_close": _clock(MARKET_CLOSE_TIME, now),

    }

def session_phase(now: datetime | None = None) -> str:

    now = now or datetime.now()

    s = session_schedule(now)

    if now < s["market_open"]:

        return "PRE_OPEN"

    if now < s["exit_window"]:

        return "OPEN"

    if now < s["forced_exit"]:

        return "EXIT_WINDOW"

    if now < s["broker_square_off"]:

        return "FORCED_EXIT"

    return "BROKER_SQUARE_OFF"

def broker_penalty_per_position() -> float:

    return BROKER_AUTO_SQUARE_OFF_PENALTY * (1 + BROKER_PENALTY_GST_PCT / 100.0)

# ---------------- portfolio ----------------

def _open_positions() -> list[dict]:

    return [dict(p) for p in (st.session_state.paper_open_positions or [])]

def _mark(position: dict) -> float:

    return float(position.get("current_price") or position["entry_price"])

def portfolio_metrics(positions: list[dict] | None = None) -> dict:

    """Leverage of what is HELD right now. Closed trades are not included."""

    positions = _open_positions() if positions is None else positions

    capital = paper_available_capital()

    exposures = [p["quantity"] * _mark(p) for p in positions]

    exposure = sum(exposures)

    margin = sum(float(p["margin_used"]) for p in positions)

    per_trade = [e / capital for e in exposures] if capital > 0 else []

    return {

        "count": len(positions),

        "exposure": exposure,

        "margin_used": margin,

        "unrealized": sum(float(p.get("unrealized", 0.0)) for p in positions),

        "leverage": (exposure / capital) if capital > 0 else 0.0,

        "avg_leverage": (sum(per_trade) / len(per_trade)) if per_trade else 0.0,

        "free_capital": max(capital - margin, 0.0),

        "capital": capital,

    }

def _update_paper_equity(unrealized: float) -> None:

    equity = paper_available_capital() + float(unrealized)

    st.session_state.paper_equity = equity

    st.session_state.paper_peak_equity = max(

        float(st.session_state.paper_peak_equity), equity

    )

    drawdown = equity - float(st.session_state.paper_peak_equity)

    st.session_state.paper_max_drawdown = min(

        float(st.session_state.paper_max_drawdown), drawdown

    )

def _refresh_paper_aggregates() -> dict:

    """Keep the summary fields used by the rest of the UI in sync."""

    positions = _open_positions()

    m = portfolio_metrics(positions)

    if not positions:

        st.session_state.paper_position = "NONE"

        st.session_state.paper_entry_price = 0.0

        st.session_state.paper_quantity = 0

        st.session_state.paper_unrealized_pnl = 0.0

    else:

        directions = {p["direction"] for p in positions}

        label = next(iter(directions)) if len(directions) == 1 else "MIXED"

        st.session_state.paper_position = (

            label if len(positions) == 1 else f"{label} ×{len(positions)}"

        )

        total_qty = sum(int(p["quantity"]) for p in positions)

        st.session_state.paper_quantity = total_qty

        st.session_state.paper_entry_price = (

            sum(p["entry_price"] * p["quantity"] for p in positions) / total_qty

            if len(directions) == 1 and total_qty > 0

            else 0.0

        )

        st.session_state.paper_unrealized_pnl = m["unrealized"]

    st.session_state.paper_portfolio_exposure = m["exposure"]

    st.session_state.paper_portfolio_leverage = m["leverage"]

    st.session_state.paper_avg_open_leverage = m["avg_leverage"]

    st.session_state.paper_margin_in_use = m["margin_used"]

    _update_paper_equity(m["unrealized"])

    return m

def _portfolio_fields() -> dict:

    m = portfolio_metrics()

    return {

        "open_positions": m["count"],

        "portfolio_exposure": m["exposure"],

        "portfolio_leverage": m["leverage"],

        "portfolio_avg_leverage": m["avg_leverage"],

        "portfolio_margin": m["margin_used"],

        "portfolio_unrealized": m["unrealized"],

    }

def _paper_exposure_snapshot(force: bool = False) -> None:

    now = datetime.now()

    last = st.session_state.paper_last_snapshot

    if not force and last:

        try:

            elapsed = (now - datetime.fromisoformat(last)).total_seconds()

            if elapsed < PAPER_EXPOSURE_SNAPSHOT_SECONDS:

                return

        except ValueError:

            pass

    m = portfolio_metrics()

    ledger_event(

        "EXPOSURE",

        {

            "direction": st.session_state.paper_position,

            "open_positions": m["count"],

            "price": st.session_state.paper_current_price,

            "exposure": m["exposure"],

            "leverage": m["leverage"],

            "avg_leverage": m["avg_leverage"],

            "margin_used": m["margin_used"],

            "unrealized_pnl": m["unrealized"],

            "equity": st.session_state.paper_equity,

        },

    )

    st.session_state.paper_last_snapshot = now.isoformat(timespec="seconds")

# ---------------- signals ----------------

def paper_record_signal(

    ensemble_signal: str,

    confidence: float,

    model_votes: dict,

    price: float,

    decision: str,

    reject_reason: str = "",

) -> None:

    """

    Call this for EVERY ensemble evaluation from the signal pipeline.

    model_votes: {"XGBoost": "LONG"/"SHORT"/"NEUTRAL", ...}

    decision: "EXECUTED", "REJECTED", "NO_ACTION" or "HOLD".

    """

    st.session_state.last_signal = ensemble_signal

    st.session_state.signal_confidence = float(confidence)

    if price > 0:

        st.session_state.current_price = float(price)

    ledger_event(

        "SIGNAL",

        {

            "ensemble_signal": str(ensemble_signal).upper(),

            "confidence": float(confidence),

            "model_votes": dict(model_votes or {}),

            "price": float(price),

            "decision": decision,

            "reject_reason": reject_reason,

        },

    )

# ---------------- open ----------------

def paper_open_position(

    direction: str,

    price: float,

    confidence: float = 0.0,

    model_votes: dict | None = None,

    signal: str | None = None,

) -> str:

    """

    Open a NEW simulated position (several can be open at once).

    Returns the trade_id, or "" if rejected (reason is in the ledger).

    """

    direction = str(direction).upper()

    def reject(reason: str) -> str:

        ledger_event(

            "TRADE_REJECTED",

            {

                "direction": direction,

                "price": float(price),

                "confidence": float(confidence),

                "reason": reason,

            },

        )

        return ""

    if direction not in ("LONG", "SHORT"):

        return reject("Invalid direction")

    if st.session_state.system_killed:

        return reject("System killed")

    if not st.session_state.paper_running:

        return reject("Paper engine not running")

    if price is None or price <= 0:

        return reject("No valid price")

    phase = session_phase()

    schedule = session_schedule()

    if phase == "PRE_OPEN":

        return reject("Market not open yet")

    if phase != "OPEN":

        return reject(

            "Exit window active — no new entries after "

            + schedule["exit_window"].strftime("%H:%M")

        )

    settings = paper_settings()

    if settings["instrument"] == "NIFTY_OPTIONS_BUY" and direction == "SHORT":

        return reject("Options-buying profile cannot short (buy a put instead)")

    capital = paper_available_capital()

    current = portfolio_metrics()

    sizing = size_position(

        settings["instrument"],

        current["free_capital"],

        float(price),

        settings["leverage"],

        settings["margin_pct"],

        settings["lot_size"],

    )

    if sizing["quantity"] <= 0:

        return reject(sizing["reason"] or "Insufficient free capital")

    # Enforce (1) the leverage cap on the WHOLE open portfolio and

    # (2) the per-trade share of that budget.

    cap = settings["max_leverage"]

    portfolio_room = cap * capital - current["exposure"]

    trade_budget = cap * capital * settings["trade_allocation_pct"] / 100.0

    room = min(portfolio_room, trade_budget)

    after = (current["exposure"] + sizing["exposure"]) / capital

    if sizing["exposure"] > room + 1e-6:

        unit = settings["lot_size"] if INSTRUMENTS[settings["instrument"]]["leverage_mode"] != "broker" else 1

        fit_qty = int(math.floor(room / (float(price) * unit))) * unit if room > 0 else 0

        if fit_qty <= 0:

            if portfolio_room < trade_budget:

                return reject(

                    f"Portfolio leverage cap {cap:.2f}× reached "

                    f"(holding {current['leverage']:.2f}×; this trade would make {after:.2f}×)"

                )

            return reject(

                f"Per-trade budget too small for one lot "

                f"({settings['trade_allocation_pct']:.0f}% of {cap:.2f}× = "

                f"₹{trade_budget:,.0f} exposure)"

            )

        scale = fit_qty / sizing["quantity"]

        sizing = {

            **sizing,

            "quantity": fit_qty,

            "lots": fit_qty // unit,

            "exposure": fit_qty * float(price),

            "margin_used": sizing["margin_used"] * scale,

        }

    votes = dict(model_votes or {})

    agreed = [name for name, vote in votes.items() if str(vote).upper() == direction]

    now = datetime.now()

    position = {

        "trade_id": uuid.uuid4().hex,

        "symbol": st.session_state.symbol,

        "instrument": settings["instrument"],

        "instrument_label": settings["instrument_label"],

        "direction": direction,

        "quantity": int(sizing["quantity"]),

        "lots": int(sizing["lots"]),

        "entry_time": now.isoformat(timespec="seconds"),

        "entry_price": float(price),

        "current_price": float(price),

        "exposure": float(sizing["exposure"]),

        "margin_used": float(sizing["margin_used"]),

        "leverage": float(sizing["exposure"]) / capital,

        "capital_base": capital,

        "signal": signal or direction,

        "confidence": float(confidence),

        "model_votes": votes,

        "models_agreed": agreed,

        "agreement_count": len(agreed),

        "unrealized": 0.0,

        "mae": 0.0,

        "mfe": 0.0,

        "slippage_pct": settings["slippage_pct"],

        "window_start_net": None,

        "window_peak_net": None,

    }

    st.session_state.paper_open_positions = _open_positions() + [position]

    st.session_state.paper_current_price = float(price)

    _refresh_paper_aggregates()

    ledger_event("TRADE_OPEN", {**position, **_portfolio_fields()})

    st.session_state.paper_last_snapshot = now.isoformat(timespec="seconds")

    update_timestamp()

    return position["trade_id"]

# ---------------- close ----------------

def _estimate_net(position: dict, price: float) -> float:

    """P&L if closed now, after slippage and charges."""

    qty = int(position["quantity"])

    entry = float(position["entry_price"])

    sign = 1 if position["direction"] == "LONG" else -1

    gross = (price - entry) * qty * sign

    slippage = (entry + price) * qty * float(position["slippage_pct"]) / 100.0

    if position["direction"] == "LONG":

        buy_value, sell_value = entry * qty, price * qty

    else:

        buy_value, sell_value = price * qty, entry * qty

    charges = compute_charges(position["instrument"], buy_value, sell_value)["total"]

    return gross - slippage - charges

def paper_close_trade(

    trade_id: str,

    price: float,

    exit_reason: str,

    penalty: float = 0.0,

) -> dict | None:

    """Close ONE open position by trade_id."""

    positions = _open_positions()

    match = next((p for p in positions if p["trade_id"] == trade_id), None)

    if match is None:

        return None

    entry = float(match["entry_price"])

    if price is None or price <= 0:

        price = _mark(match)

    price = float(price)

    quantity = int(match["quantity"])

    direction = match["direction"]

    sign = 1 if direction == "LONG" else -1

    gross = (price - entry) * quantity * sign

    slippage = (entry + price) * quantity * float(match["slippage_pct"]) / 100.0

    if direction == "LONG":

        buy_value, sell_value = entry * quantity, price * quantity

    else:

        buy_value, sell_value = price * quantity, entry * quantity

    charges = compute_charges(match["instrument"], buy_value, sell_value)

    penalty = float(penalty)

    net = gross - slippage - charges["total"] - penalty

    now = datetime.now()

    try:

        holding = (now - datetime.fromisoformat(match["entry_time"])).total_seconds() / 60.0

    except Exception:

        holding = 0.0

    record = {

        **match,

        "exit_time": now.isoformat(timespec="seconds"),

        "exit_price": price,

        "gross_pnl": round(gross, 2),

        "slippage": round(slippage, 2),

        "charges": charges,

        "charges_total": charges["total"],

        "brokerage": charges["brokerage"],

        "taxes_fees": charges["taxes_fees"],

        "penalty": round(penalty, 2),

        "net_pnl": round(net, 2),

        "holding_minutes": round(holding, 2),

        "exit_reason": exit_reason,

        "mae": round(min(float(match.get("mae", 0.0)), gross), 2),

        "mfe": round(max(float(match.get("mfe", 0.0)), gross), 2),

    }

    # Remove it FIRST so it no longer counts in leverage.

    st.session_state.paper_open_positions = [

        p for p in positions if p["trade_id"] != trade_id

    ]

    st.session_state.paper_realized_pnl += net

    st.session_state.paper_gross_pnl += gross

    st.session_state.paper_charges += charges["total"]

    st.session_state.paper_slippage += slippage

    st.session_state.paper_penalties += penalty

    st.session_state.paper_net_pnl += net

    st.session_state.paper_trade_count += 1

    if net > 0:

        st.session_state.paper_winning_trades += 1

    else:

        st.session_state.paper_losing_trades += 1

    st.session_state.paper_trade_log = list(st.session_state.paper_trade_log) + [

        {

            "Time": now.strftime("%Y-%m-%d %H:%M:%S"),

            "Mode": "PAPER",

            "Symbol": match["symbol"],

            "Action": exit_reason,

            "Position": direction,

            "Qty": quantity,

            "Entry": entry,

            "Exit": price,

            "Leverage": round(float(match["leverage"]), 2),

            "Gross P&L": round(gross, 2),

            "Charges": charges["total"],

            "Slippage": round(slippage, 2),

            "Penalty": round(penalty, 2),

            "Net P&L": round(net, 2),

        }

    ]

    _refresh_paper_aggregates()

    ledger_event("TRADE_CLOSE", {**record, **_portfolio_fields()})

    _paper_exposure_snapshot(force=True)

    update_timestamp()

    return record

def paper_close_all(

    price: float,

    exit_reason: str,

    penalty_per_position: float = 0.0,

) -> list[dict]:

    records = []

    for position in _open_positions():

        record = paper_close_trade(

            position["trade_id"], price, exit_reason, penalty_per_position

        )

        if record:

            records.append(record)

    return records

def paper_close_position(price: float, exit_reason: str) -> list[dict]:

    """Backward-compatible: closes ALL open positions."""

    return paper_close_all(price, exit_reason)

# ---------------- session close rules ----------------

def apply_session_close_rules(price: float, now: datetime | None = None) -> None:

    positions = _open_positions()

    if not positions:

        return

    now = now or datetime.now()

    if price is None or price <= 0:

        price = _mark(positions[0])

    # Intraday positions left from a previous day: broker squares off.

    stale = [

        p for p in positions

        if datetime.fromisoformat(p["entry_time"]).date() < now.date()

    ]

    for p in stale:

        paper_close_trade(

            p["trade_id"], _mark(p), "BROKER_AUTO_SQUARE_OFF",

            broker_penalty_per_position(),

        )

    if stale:

        positions = _open_positions()

        if not positions:

            return

    phase = session_phase(now)

    if phase == "BROKER_SQUARE_OFF":

        paper_close_all(price, "BROKER_AUTO_SQUARE_OFF", broker_penalty_per_position())

        return

    if phase == "FORCED_EXIT":

        paper_close_all(price, "PRE_CLOSE_EXIT")

        return

    if phase != "EXIT_WINDOW":

        return

    updated = []

    to_close = []

    for p in positions:

        est = _estimate_net(p, price)

        if p.get("window_start_net") is None:

            p["window_start_net"] = est

            p["window_peak_net"] = est

        p["window_peak_net"] = max(float(p["window_peak_net"]), est)

        start = float(p["window_start_net"])

        peak = float(p["window_peak_net"])

        exposure_now = p["quantity"] * price

        reason = None

        if start < 0 <= est:

            reason = "WINDOW_BREAKEVEN_EXIT"

        elif est > 0 and peak > 0 and est <= peak * (1 - EXIT_TRAIL_GIVEBACK_PCT / 100.0):

            reason = "WINDOW_TRAIL_EXIT"

        elif est < 0 and est <= start - exposure_now * EXIT_LOSS_EXTENSION_PCT / 100.0:

            reason = "WINDOW_STOP_EXIT"

        updated.append(p)

        if reason:

            to_close.append((p["trade_id"], reason))

    st.session_state.paper_open_positions = updated

    for trade_id, reason in to_close:

        paper_close_trade(trade_id, price, reason)

def paper_mark_to_market(price: float) -> None:

    """Update every open position, portfolio leverage, equity and close rules."""

    if price is None or price <= 0:

        return

    price = float(price)

    st.session_state.paper_current_price = price

    updated = []

    for p in _open_positions():

        sign = 1 if p["direction"] == "LONG" else -1

        unrealized = (price - float(p["entry_price"])) * int(p["quantity"]) * sign

        p["current_price"] = price

        p["unrealized"] = unrealized

        p["mae"] = min(float(p.get("mae", 0.0)), unrealized)

        p["mfe"] = max(float(p.get("mfe", 0.0)), unrealized)

        updated.append(p)

    st.session_state.paper_open_positions = updated

    _refresh_paper_aggregates()

    if updated:

        _paper_exposure_snapshot()

    apply_session_close_rules(price)


# ============================================================

# SHOONYA AUTHENTICATION

# ============================================================

def add_shoonya_message(level: str, text: str) -> None:

    """Store a connection message so it survives st.rerun()."""

    messages = list(st.session_state.get("shoonya_messages", []))

    messages.append((level, text))

    st.session_state.shoonya_messages = messages

def clear_shoonya_messages() -> None:

    st.session_state.shoonya_messages = []

def _is_valid_url(url: str, schemes: set[str]) -> bool:

    try:

        parsed = urlparse(url)

    except Exception:

        return False

    return parsed.scheme in schemes and bool(parsed.netloc)

def validate_shoonya_urls() -> None:

    if not _is_valid_url(OAUTH_URL, {"https"}):

        raise RuntimeError(f"OAUTH_URL is not a valid https URL: {OAUTH_URL!r}")

    if not _is_valid_url(REST_HOST, {"https"}):

        raise RuntimeError(f"REST_HOST is not a valid https URL: {REST_HOST!r}")

    if not _is_valid_url(WEBSOCKET_URL, {"wss"}):

        raise RuntimeError(

            f"WEBSOCKET_URL is not a valid wss URL: {WEBSOCKET_URL!r}"

        )

def _secret_lookup_keys() -> list[str]:

    """Credential Manager usernames to try, in order."""

    keys = []

    candidates = [CLIENT_ID, USER_ID]

    if CLIENT_ID.endswith("_U"):

        candidates.append(CLIENT_ID[:-2])

    for key in candidates:

        if key and key not in keys:

            keys.append(key)

    return keys

def get_secret_code_with_key() -> tuple[str | None, str | None]:

    """Return (secret, credential_username_used). Whitespace is stripped."""

    for key in _secret_lookup_keys():

        try:

            secret = keyring.get_password(KEYRING_SERVICE, key)

        except Exception:

            secret = None

        if secret and secret.strip():

            return secret.strip(), key

    return None, None

def get_secret_code() -> str | None:

    return get_secret_code_with_key()[0]

def create_shoonya_api() -> NorenApi:

    return NorenApi(

        host=REST_HOST,

        websocket=WEBSOCKET_URL,

    )

def _parse_token_result(result):

    """

    Normalise the getAccessToken() response.

    Accepts a tuple/list (access_token, uid, refresh_token, account_id),

    a dict, or a bare token string.

    """

    if result is None:

        raise RuntimeError("Shoonya returned no response to the token request.")

    if isinstance(result, dict):

        if str(result.get("stat", "")).lower() == "not_ok":

            raise RuntimeError(

                f"Shoonya rejected token request: "

                f"{result.get('emsg', result)}"

            )

        access_token = (

            result.get("access_token")

            or result.get("susertoken")

            or result.get("token")

        )

        returned_uid = result.get("uid") or result.get("UID") or USER_ID

        refresh_token = result.get("refresh_token")

        account_id = result.get("actid") or result.get("account_id") or returned_uid

        return access_token, returned_uid, refresh_token, account_id

    if isinstance(result, (tuple, list)):

        items = list(result) + [None, None, None, None]

        access_token, returned_uid, refresh_token, account_id = items[:4]

        return (

            access_token,

            returned_uid or USER_ID,

            refresh_token,

            account_id or returned_uid or USER_ID,

        )

    if isinstance(result, str) and result:

        return result, USER_ID, None, USER_ID

    raise RuntimeError(

        f"Unexpected Shoonya SDK response type: {type(result).__name__}"

    )

def _apply_oauth_session(api, access_token, uid, account_id) -> None:

    """Attach the OAuth session to the API object when the SDK supports it."""

    inject = getattr(api, "injectOAuthHeader", None)

    if callable(inject):

        try:

            inject(access_token, uid, account_id)

            return

        except TypeError:

            inject(access_token, uid)

            return

# Parameter-name aliases used to match getAccessToken() arguments

# by NAME, so the call is correct whatever order the SDK version uses.

_TOKEN_PARAM_ALIASES = {

    "code": {"code", "auth_code", "authcode", "authorization_code", "oauth_code"},

    "secret": {

        "secret", "secret_code", "secretcode", "secret_key", "secretkey",

        "api_secret", "apisecret", "app_secret", "appsecret",

    },

    "client_id": {

        "client_id", "clientid", "client", "app_key", "appkey",

        "api_key", "apikey", "vendor_code", "vendorcode",

    },

    "uid": {"uid", "userid", "user_id", "user", "username"},

}

def _token_call_plan(func, auth_code, secret, client_id, uid):

    """

    Work out how to call getAccessToken() for the installed SDK.

    Returns (args, kwargs, description).

    """

    values = {

        "code": auth_code,

        "secret": secret,

        "client_id": client_id,

        "uid": uid,

    }

    default_order = [auth_code, secret, client_id, uid]

    try:

        params = [

            p

            for p in inspect.signature(func).parameters.values()

            if p.name != "self"

            and p.kind

            in (

                inspect.Parameter.POSITIONAL_OR_KEYWORD,

                inspect.Parameter.KEYWORD_ONLY,

            )

        ]

    except (TypeError, ValueError):

        return tuple(default_order), {}, "signature unavailable; default order"

    kwargs = {}

    used_roles = set()

    mapping = []

    for param in params:

        lowered = param.name.lower()

        for role, aliases in _TOKEN_PARAM_ALIASES.items():

            if role in used_roles:

                continue

            if lowered in aliases:

                kwargs[param.name] = values[role]

                used_roles.add(role)

                mapping.append(f"{param.name}={role}")

                break

    required = [p for p in params if p.default is inspect.Parameter.empty]

    if kwargs and all(p.name in kwargs for p in required):

        return (), kwargs, "matched by name: " + ", ".join(mapping)

    count = min(len(params), 4) if params else 4

    return (

        tuple(default_order[:count]),

        {},

        f"names not recognised; positional (code, secret, client_id, uid)[:{count}]",

    )

def _redact(text: str, *secrets) -> str:

    for value in secrets:

        if value and isinstance(value, str) and len(value) >= 4:

            text = text.replace(value, "XXXX")

    return text

def exchange_oauth_code(auth_code: str):

    secret_code, secret_key = get_secret_code_with_key()

    if not secret_code:

        raise RuntimeError(

            "Shoonya Secret Code was not found in Windows Credential Manager "

            f"(service '{KEYRING_SERVICE}', tried usernames: "

            f"{', '.join(_secret_lookup_keys())})."

        )

    api = create_shoonya_api()

    args, kwargs, plan = _token_call_plan(

        api.getAccessToken, auth_code, secret_code, CLIENT_ID, USER_ID

    )

    # Capture SDK logging AND prints, so a rejection shows Shoonya's reason.

    log_buffer = io.StringIO()

    handler = logging.StreamHandler(log_buffer)

    handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()

    previous_level = root_logger.level

    root_logger.addHandler(handler)

    root_logger.setLevel(logging.DEBUG)

    call_error = None

    result = None

    try:

        with contextlib.redirect_stdout(log_buffer):

            result = api.getAccessToken(*args, **kwargs)

    except Exception as exc:

        call_error = exc

    finally:

        root_logger.removeHandler(handler)

        root_logger.setLevel(previous_level)

    sdk_log = _redact(

        log_buffer.getvalue().strip(), secret_code, auth_code

    )

    details = (

        f"Call plan: {plan}. "

        f"Secret read from Credential Manager username '{secret_key}' "

        f"(length {len(secret_code)}). "

        f"Auth code length: {len(auth_code)}. "

        "SDK log: " + (sdk_log[-1500:] if sdk_log else "(empty)")

    )

    if call_error is not None:

        raise RuntimeError(

            f"getAccessToken() raised {type(call_error).__name__}: "

            f"{_redact(str(call_error), secret_code, auth_code)}. {details}"

        )

    if result is None:

        raise RuntimeError("Shoonya rejected the token request. " + details)

    return api, result

def _extract_auth_code(url: str) -> str | None:

    """Extract the OAuth ?code= value (query or fragment), URL-decoded."""

    if not url or "code=" not in url:

        return None

    try:

        parsed = urlparse(url)

        for part in (parsed.query, parsed.fragment):

            if not part:

                continue

            values = parse_qs(part).get("code")

            if values and values[0]:

                return values[0]

    except Exception:

        pass

    code_part = url.split("code=", 1)[1]

    code = code_part.split("&", 1)[0].split("#", 1)[0]

    return code or None

def oauth_login_url() -> str:

    return f"{OAUTH_URL}?client_id={CLIENT_ID}"

def capture_oauth_code_with_browser():

    """

    Opens Chrome, lets the user complete Shoonya login/OTP manually,

    and captures the OAuth authorization code from the redirect URL.

    """

    if not CLIENT_ID:

        raise RuntimeError("SHOONYA_CLIENT_ID is not configured.")

    validate_shoonya_urls()

    options = Options()

    options.add_argument("--start-maximized")

    options.add_argument("--disable-notifications")

    driver = webdriver.Chrome(options=options)

    st.session_state.oauth_browser_active = True

    st.session_state.shoonya_status = "WAITING FOR OTP"

    update_timestamp()

    start_time = time.time()

    try:

        driver.get(oauth_login_url())

        while time.time() - start_time < OAUTH_TIMEOUT_SECONDS:

            try:

                current_url = driver.current_url

            except (NoSuchWindowException, WebDriverException):

                raise RuntimeError(

                    "The Shoonya login browser was closed before "

                    "login completed."

                )

            auth_code = _extract_auth_code(current_url)

            if auth_code:

                return auth_code

            time.sleep(OAUTH_POLL_INTERVAL_SECONDS)

        raise TimeoutError("OAuth login timed out after 5 minutes.")

    finally:

        try:

            driver.quit()

        except Exception:

            pass

def _reset_shoonya_session_state() -> None:

    st.session_state.shoonya_session = None

    st.session_state.shoonya_access_token = None

    st.session_state.shoonya_refresh_token = None

    st.session_state.oauth_code = None

    st.session_state.oauth_browser_active = False

def _preflight_checks() -> bool:

    if not CLIENT_ID:

        add_shoonya_message("error", "Shoonya Client ID is missing.")

        return False

    if not get_secret_code():

        add_shoonya_message(

            "error",

            "Shoonya Secret Code is not stored in Windows Credential Manager "

            f"(service '{KEYRING_SERVICE}', tried usernames: "

            f"{', '.join(_secret_lookup_keys())}).",

        )

        return False

    try:

        validate_shoonya_urls()

    except Exception as exc:

        add_shoonya_message("error", str(exc))

        return False

    return True

def _complete_shoonya_connection(auth_code: str) -> None:

    """Exchange an OAuth code for a session and start the market feed."""

    st.session_state.shoonya_status = "AUTHENTICATING"

    with st.spinner("Exchanging OAuth code for Shoonya session..."):

        api, result = exchange_oauth_code(auth_code)

    access_token, returned_uid, refresh_token, account_id = (

        _parse_token_result(result)

    )

    if not access_token:

        raise RuntimeError("Shoonya did not return an access token.")

    _apply_oauth_session(api, access_token, returned_uid, account_id)

    st.session_state.shoonya_session = api

    st.session_state.shoonya_access_token = access_token

    st.session_state.shoonya_refresh_token = refresh_token

    st.session_state.oauth_code = None

    st.session_state.oauth_browser_active = False

    st.session_state.shoonya_status = "CONNECTED"

    st.session_state.data_status = "SESSION READY"

    persist_to_shared_state()

    start_market_feed(api)

    update_timestamp()

    add_shoonya_message("success", "Shoonya authentication successful.")

    add_shoonya_message(

        "info",

        f"Connected UID: {returned_uid} | Account: {account_id}",

    )

def _handle_connection_failure(exc: Exception) -> None:

    _reset_shoonya_session_state()

    st.session_state.shoonya_status = "ERROR"

    update_timestamp()

    message = str(exc).split("Stacktrace:", 1)[0].strip()

    add_shoonya_message("error", f"Shoonya connection failed: {message}")

def connect_shoonya() -> None:

    """Automatic flow: Selenium Chrome captures the redirect code."""

    clear_shoonya_messages()

    if not _preflight_checks():

        return

    stop_market_feed()

    st.session_state.shoonya_status = "AUTHENTICATING"

    update_timestamp()

    try:

        with st.spinner(

            "Opening Shoonya login. Complete login and OTP in the browser..."

        ):

            auth_code = capture_oauth_code_with_browser()

        st.session_state.oauth_code = auth_code

        _complete_shoonya_connection(auth_code)

    except Exception as exc:

        _handle_connection_failure(exc)

def open_shoonya_login_in_normal_browser() -> None:

    """Manual flow step 1: open the login page in the normal browser."""

    clear_shoonya_messages()

    if not _preflight_checks():

        return

    webbrowser.open(oauth_login_url())

    add_shoonya_message(

        "info",

        "Shoonya login opened in your normal browser. After OTP, copy the "

        "FULL address-bar URL of the page you land on and paste it below.",

    )

def connect_shoonya_manual(pasted: str) -> None:

    """Manual flow step 2: exchange a pasted redirect URL or code."""

    clear_shoonya_messages()

    if not _preflight_checks():

        return

    pasted = (pasted or "").strip()

    auth_code = _extract_auth_code(pasted)

    if not auth_code and pasted and "/" not in pasted and "=" not in pasted:

        auth_code = pasted  # user pasted just the code

    if not auth_code:

        add_shoonya_message(

            "error",

            "No 'code=' found. Paste the full URL from the address bar "

            "after login, or just the code value.",

        )

        return

    stop_market_feed()

    update_timestamp()

    try:

        _complete_shoonya_connection(auth_code)

    except Exception as exc:

        _handle_connection_failure(exc)

def run_shoonya_diagnostics() -> None:

    """Checks the setup WITHOUT logging in (uses no OTP)."""

    clear_shoonya_messages()

    lines = []

    lines.append(f"Client ID: {CLIENT_ID or 'MISSING'}")

    lines.append(f"User ID: {USER_ID or 'MISSING'}")

    secret, secret_key = get_secret_code_with_key()

    if secret:

        lines.append(

            f"Secret Code: FOUND under username '{secret_key}' "

            f"(length {len(secret)})"

        )

    else:

        lines.append(

            "Secret Code: NOT FOUND (tried usernames: "

            f"{', '.join(_secret_lookup_keys())})"

        )

    try:

        validate_shoonya_urls()

        lines.append("URLs: valid")

    except Exception as exc:

        lines.append(f"URLs: INVALID — {exc}")

    try:

        api = create_shoonya_api()

        func = api.getAccessToken

        lines.append(f"getAccessToken signature: {inspect.signature(func)}")

        _, _, plan = _token_call_plan(func, "CODE", "SECRET", CLIENT_ID, USER_ID)

        lines.append(f"Token call plan: {plan}")

        lines.append(

            "injectOAuthHeader available: "

            + ("YES" if callable(getattr(api, "injectOAuthHeader", None)) else "NO")

        )

        try:

            source = inspect.getsource(type(api).getAccessToken)

            lines.append("\ngetAccessToken source:\n" + source)

        except Exception:

            lines.append("getAccessToken source: unavailable")

    except Exception as exc:

        lines.append(f"SDK check failed: {exc}")

    st.session_state.shoonya_diag_text = "\n".join(lines)

    add_shoonya_message("info", "Setup check complete (no login used).")

def disconnect_shoonya() -> None:

    stop_market_feed()

    _reset_shoonya_session_state()

    st.session_state.shoonya_status = "NOT CONNECTED"

    st.session_state.data_status = "NOT CONNECTED"

    clear_shoonya_messages()

    add_shoonya_message("info", "Shoonya disconnected.")

    update_timestamp()

# ============================================================

# TRADING MODE

# ============================================================

def switch_mode(mode: str) -> None:

    # Never allow switching while a paper position is active.

    if st.session_state.paper_position != "NONE":

        st.warning(

            "Close the active paper position before changing trading mode."

        )

        return

    # Never allow a real mode switch while real executor is running.

    if st.session_state.real_running:

        st.warning(

            "Real execution is active. Stop the real executor first."

        )

        return

    st.session_state.trading_mode = mode

    if mode == "REAL":

        st.session_state.real_unlock_confirmed = False

    update_timestamp()

def start_active_engine() -> None:

    if st.session_state.system_killed:

        st.warning("System is killed. Reset the kill switch first.")

        return

    if st.session_state.shoonya_status != "CONNECTED":

        st.warning("Connect Shoonya before starting the trading engine.")

        return

    if active_mode() == "PAPER":

        st.session_state.paper_running = True

        settings = paper_settings()

        ledger_event(

            "ENGINE_START",

            {

                **settings,

                "capital": paper_available_capital(),

                "symbol": st.session_state.symbol,

            },

        )

        update_timestamp()

        return

    # REAL mode is intentionally locked in V3.

    st.error(

        "REAL execution is LOCKED in this build. "

        "No real-money order can be submitted."

    )

def stop_active_engine() -> None:

    if active_mode() == "PAPER":

        st.session_state.paper_running = False

        ledger_event(

            "ENGINE_STOP",

            {"open_position": st.session_state.paper_position},

        )

    else:

        st.session_state.real_running = False

    update_timestamp()

def emergency_kill() -> None:

    st.session_state.paper_running = False

    st.session_state.real_running = False

    st.session_state.system_killed = True

    st.session_state.real_unlock_confirmed = False

    ledger_event(

        "KILL_SWITCH",

        {"open_position": st.session_state.paper_position},

    )

    if st.session_state.paper_position != "NONE":

        # Close at the last known price, with charges and slippage.

        paper_close_position(

            st.session_state.paper_current_price

            or st.session_state.paper_entry_price,

            "MANUAL_KILL_SWITCH",

        )

    # Real mode deliberately does NOT attempt an order here because

    # real order submission is disabled in this version.

    update_timestamp()

def clear_kill_switch() -> None:

    st.session_state.system_killed = False

    ledger_event("KILL_RESET", {})

    update_timestamp()


# ============================================================

# UI HEADER

# ============================================================

st.title("📈 Athena 3.0")

st.subheader("Unified Trading Terminal")

mode = st.radio(

    "Trading Mode",

    ["PAPER", "REAL"],

    index=0 if st.session_state.trading_mode == "PAPER" else 1,

    horizontal=True,

    key="trading_mode_radio",

)

if mode != st.session_state.trading_mode:

    switch_mode(mode)

    st.rerun()

if st.session_state.trading_mode == "PAPER":

    st.success(

        "🟢 PAPER MODE — simulated execution only. "

        "No real-money orders are submitted."

    )

else:

    st.error(

        "🔴 REAL MODE — BROKER EXECUTION LOCKED. "

        "This build cannot submit real-money orders."

    )

if st.session_state.system_killed:

    st.error(

        "🛑 SYSTEM KILLED — all trading engines are frozen. "

        "Reset the kill switch before starting again."

    )

# ============================================================

# SHOONYA CONNECTION

# ============================================================

st.divider()

st.header("🔐 Shoonya Connection")

broker_cols = st.columns(4)

with broker_cols[0]:

    status_badge("Shoonya", st.session_state.shoonya_status)

with broker_cols[1]:

    st.metric("Client ID", CLIENT_ID if CLIENT_ID else "MISSING")

with broker_cols[2]:

    st.metric("User ID", USER_ID if USER_ID else "MISSING")

with broker_cols[3]:

    st.metric(

        "Session",

        "READY" if st.session_state.shoonya_session else "NOT READY",

    )

broker_button_cols = st.columns(2)

with broker_button_cols[0]:

    if st.button(

        "🔐 CONNECT SHOONYA",

        type="primary",

        use_container_width=True,

        disabled=st.session_state.shoonya_status

        in {"AUTHENTICATING", "WAITING FOR OTP"},

    ):

        connect_shoonya()

        # Refresh the page after authentication so the header/session

        # indicators reflect the newly established Shoonya session.

        persist_to_shared_state()

        st.rerun()

with broker_button_cols[1]:

    if st.button(

        "🔌 DISCONNECT SHOONYA",

        use_container_width=True,

        disabled=st.session_state.shoonya_session is None,

    ):

        disconnect_shoonya()

        persist_to_shared_state()

        st.rerun()

# FIX: show connection results that would otherwise be lost on rerun.

for _level, _text in st.session_state.get("shoonya_messages", []):

    if _level == "success":

        st.success(_text)

    elif _level == "error":

        st.error(_text)

    elif _level == "warning":

        st.warning(_text)

    else:

        st.info(_text)

if st.session_state.get("shoonya_diag_text"):

    with st.expander("Shoonya setup check results", expanded=True):

        st.code(st.session_state.shoonya_diag_text, language="text")

with st.expander("🛠️ Shoonya tools: setup check & manual login (no Selenium)"):

    st.caption(

        "Setup check uses NO login and NO OTP. Manual login opens your "

        "normal browser; after OTP, paste the full redirect URL here."

    )

    tool_cols = st.columns(2)

    with tool_cols[0]:

        if st.button("🔎 CHECK SHOONYA SETUP (no OTP)", use_container_width=True):

            run_shoonya_diagnostics()

            persist_to_shared_state()

            st.rerun()

    with tool_cols[1]:

        if st.button("🌐 OPEN LOGIN IN NORMAL BROWSER", use_container_width=True):

            open_shoonya_login_in_normal_browser()

            persist_to_shared_state()

            st.rerun()

    pasted_redirect = st.text_input(

        "Paste redirect URL (or just the code) after login",

        key="shoonya_pasted_redirect",

    )

    if st.button(

        "✅ CONNECT WITH PASTED CODE",

        use_container_width=True,

        disabled=not pasted_redirect.strip(),

    ):

        connect_shoonya_manual(pasted_redirect)

        persist_to_shared_state()

        st.rerun()

if st.session_state.shoonya_status == "WAITING FOR OTP":

    st.info(

        "Shoonya login is open in Chrome. Complete login and enter OTP "

        "manually. Athena will capture the OAuth code automatically."

    )

# ============================================================

# ACCOUNT / MODE SUMMARY

# ============================================================

st.divider()

st.header("💼 Account & Mode")

summary_cols = st.columns(4)

with summary_cols[0]:

    if active_mode() == "PAPER":

        st.metric(
            "Paper Capital",
            format_rupees(float(st.session_state.paper_capital)),
        )

    else:

        st.metric("Real Account", "Shoonya")

with summary_cols[1]:

    if active_mode() == "PAPER":

        _pm = portfolio_metrics()

        st.metric(

            "Portfolio Leverage (open)",

            f"{_pm['leverage']:.2f}×",

            help=(

                "Total exposure of positions currently held ÷ capital. "

                "Closed positions are not included."

            ),

        )

    else:

        st.metric("Real Orders", "LOCKED")

with summary_cols[2]:

    st.metric("Mode", active_mode())

with summary_cols[3]:

    st.metric("Symbol", st.session_state.symbol)

# ============================================================

# PAPER RISK & LEVERAGE SETTINGS

# ============================================================

def _feed_ltp() -> float:

    try:

        return float(get_market_feed_snapshot()["quote"].get("lp", 0) or 0)

    except (TypeError, ValueError):

        return 0.0

if active_mode() == "PAPER":

    _ps = paper_settings()

    _profile = INSTRUMENTS[_ps["instrument"]]

    _sched = session_schedule()

    st.info(

        f"Instrument: **{_ps['instrument_label']}** · Leverage cap enforced on "

        f"open positions: **{_ps['max_leverage']:.2f}×** · Holding now: "

        f"**{st.session_state.paper_portfolio_leverage:.2f}×** "

        f"({len(st.session_state.paper_open_positions)} open). "

        f"Entries stop {_sched['exit_window']:%H:%M}, all flat by "

        f"{_sched['forced_exit']:%H:%M}, broker square-off "

        f"{_sched['broker_square_off']:%H:%M} (penalty)."

    )

    _ltp = _feed_ltp()

    if _ltp > 0 and _ps["instrument"] != "NIFTY_OPTIONS_BUY":

        _check = size_position(

            _ps["instrument"],

            paper_available_capital(),

            _ltp,

            _ps["leverage"],

            _ps["margin_pct"],

            _ps["lot_size"],

        )

        if _check["quantity"] <= 0:

            st.warning(f"⚠️ At NIFTY {_ltp:,.2f}: {_check['reason']}")

        else:

            st.caption(

                f"At NIFTY {_ltp:,.2f}: can open {_check['lots']} lot(s) / "

                f"{_check['quantity']} qty — exposure "

                f"{format_rupees(_check['exposure'])}, margin "

                f"{format_rupees(_check['margin_used'])}, "

                f"leverage {_check['leverage']:.2f}×."

            )

    _locked = (

        st.session_state.paper_running

        or st.session_state.paper_position != "NONE"

    )

    with st.expander("⚙️ Paper Risk & Leverage Settings", expanded=False):

        if _locked:

            st.caption("Stop the paper engine and close positions to change settings.")

        _keys = list(INSTRUMENTS.keys())

        _new_instrument = st.selectbox(

            "Instrument",

            _keys,

            index=_keys.index(st.session_state.paper_instrument),

            format_func=instrument_label,

            disabled=_locked,

            key="ui_paper_instrument",

        )

        _new_profile = INSTRUMENTS[_new_instrument]

        _new_capital = st.number_input(

            "Paper capital (₹)",

            min_value=1_000.0,

            step=1_000.0,

            value=float(st.session_state.paper_capital),

            disabled=_locked,

            key="ui_paper_capital",

        )

        _new_leverage = float(st.session_state.paper_leverage)

        _new_margin = float(st.session_state.paper_margin_pct)

        _new_lot = int(st.session_state.paper_lot_size)

        if _new_profile["leverage_mode"] == "broker":

            _low, _high = leverage_bounds(_new_instrument)

            _new_leverage = st.slider(

                "Intraday leverage (×)",

                min_value=_low,

                max_value=_high,

                value=min(max(_new_leverage, _low), _high),

                step=0.5,

                disabled=_locked,

                key="ui_paper_leverage",

            )

            _new_lot = 1

        elif _new_profile["leverage_mode"] == "exchange_margin":

            _new_margin = st.number_input(

                "Exchange margin % of contract value (SPAN + exposure)",

                min_value=float(_new_profile["min_margin_pct"]),

                max_value=float(_new_profile["max_margin_pct"]),

                step=0.5,

                value=_new_margin,

                disabled=_locked,

                key="ui_paper_margin",

            )

            _new_lot = int(st.number_input(

                "Lot size",

                min_value=1,

                step=1,

                value=int(_new_profile["lot_size"]) if _new_instrument != st.session_state.paper_instrument else _new_lot,

                disabled=_locked,

                key="ui_paper_lot_fut",

            ))

            _new_leverage = 1.0

        else:

            _new_lot = int(st.number_input(

                "Lot size",

                min_value=1,

                step=1,

                value=int(_new_profile["lot_size"]) if _new_instrument != st.session_state.paper_instrument else _new_lot,

                disabled=_locked,

                key="ui_paper_lot_opt",

            ))

            _new_leverage = 1.0

        _ceiling = effective_max_leverage(_new_instrument, _new_leverage, _new_margin)

        _new_cap = st.number_input(

            "Portfolio leverage cap (×) — total of all open positions",

            min_value=1.0,

            max_value=float(max(1.0, round(_ceiling, 2))),

            step=0.5,

            value=float(min(st.session_state.paper_leverage_cap, max(1.0, round(_ceiling, 2)))),

            disabled=_locked,

            key="ui_paper_leverage_cap",

        )

        _new_alloc = st.number_input(

            "Max share of leverage budget per trade (%)",

            min_value=5.0,

            max_value=100.0,

            step=5.0,

            value=float(st.session_state.paper_trade_allocation_pct),

            disabled=_locked,

            key="ui_paper_trade_alloc",

            help="100% = one trade can use the whole cap. 50% = at least two positions can be held.",

        )

        _new_slippage = st.number_input(

            "Slippage per side (%)",

            min_value=0.0,

            max_value=1.0,

            step=0.005,

            format="%.3f",

            value=float(st.session_state.paper_slippage_pct),

            disabled=_locked,

            key="ui_paper_slippage",

        )

        st.caption(_new_profile["note"])

        st.caption(

            "Charge rates, margin % and lot size are defaults — verify with "

            "Shoonya and edit athena_reporting/instrument_profiles.py if needed."

        )

        _set_cols = st.columns(2)

        with _set_cols[0]:

            if st.button(

                "💾 APPLY PAPER SETTINGS",

                use_container_width=True,

                disabled=_locked,

            ):

                st.session_state.paper_instrument = _new_instrument

                st.session_state.paper_capital = float(_new_capital)

                st.session_state.paper_leverage = float(_new_leverage)

                st.session_state.paper_margin_pct = float(_new_margin)

                st.session_state.paper_lot_size = int(_new_lot)

                st.session_state.paper_slippage_pct = float(_new_slippage)

                st.session_state.paper_leverage_cap = float(_new_cap)

                st.session_state.paper_trade_allocation_pct = float(_new_alloc)

                if st.session_state.paper_trade_count == 0:

                    st.session_state.paper_equity = float(_new_capital)

                    st.session_state.paper_peak_equity = float(_new_capital)

                ledger_event("SETTINGS", paper_settings())

                update_timestamp()

                persist_to_shared_state()

                st.rerun()

        with _set_cols[1]:

            if st.button(

                "♻️ RESET PAPER ACCOUNT",

                use_container_width=True,

                disabled=_locked,

            ):

                reset_paper_account()

                persist_to_shared_state()

                st.rerun()

if st.session_state.get("ledger_error"):

    st.error(st.session_state.ledger_error)

# ============================================================

# REAL MODE SAFETY GATE

# ============================================================

if active_mode() == "REAL":

    st.warning(

        "Real mode is currently an interface/safety state only. "

        "The real broker executor is intentionally disabled."

    )

    confirm = st.checkbox(

        "I understand that REAL mode is selected, but real order submission "

        "is currently locked.",

        value=st.session_state.real_unlock_confirmed,

    )

    st.session_state.real_unlock_confirmed = confirm

    if not REAL_ORDER_EXECUTION_ENABLED:

        st.info(

            "🔒 REAL EXECUTOR LOCKED — enabling actual order submission "

            "will be a separate controlled engineering step."

        )

# ============================================================

# TOP STATUS

# ============================================================

st.divider()

st.header("📊 Market & Model Status")

status_cols = st.columns(4)

with status_cols[0]:

    status_badge("Market Data", st.session_state.data_status)

with status_cols[1]:

    status_badge("Model", st.session_state.model_status)

with status_cols[2]:

    engine_state = (

        "RUNNING"

        if (

            st.session_state.paper_running

            if active_mode() == "PAPER"

            else st.session_state.real_running

        )

        else "STOPPED"

    )

    status_badge("Engine", engine_state)

with status_cols[3]:

    st.metric("Last Update", st.session_state.last_update)

# ============================================================

# TRADING CONTROLS

# ============================================================

st.divider()

st.header("🎛️ Trading Controls")

control_cols = st.columns(4)

with control_cols[0]:

    if st.button(

        f"▶ START {active_mode()}",

        type="primary",

        use_container_width=True,

        disabled=(

            st.session_state.system_killed

            or (

                st.session_state.paper_running

                if active_mode() == "PAPER"

                else st.session_state.real_running

            )

        ),

    ):

        start_active_engine()

        persist_to_shared_state()

        st.rerun()

with control_cols[1]:

    if st.button(

        f"⏹ STOP {active_mode()}",

        use_container_width=True,

        disabled=not (

            st.session_state.paper_running

            if active_mode() == "PAPER"

            else st.session_state.real_running

        ),

    ):

        stop_active_engine()

        persist_to_shared_state()

        st.rerun()

with control_cols[2]:

    if st.button(

        "🛑 EMERGENCY KILL",

        type="primary",

        use_container_width=True,

    ):

        emergency_kill()

        persist_to_shared_state()

        st.rerun()

with control_cols[3]:

    if st.button(

        "🔄 RESET KILL SWITCH",

        use_container_width=True,

        disabled=not st.session_state.system_killed,

    ):

        clear_kill_switch()

        persist_to_shared_state()

        st.rerun()

# ============================================================

# SIGNAL PANEL

# ============================================================

st.divider()

st.header("🧠 Current Model Signal")

signal_cols = st.columns(4)

with signal_cols[0]:

    st.metric("Signal", st.session_state.last_signal)

with signal_cols[1]:

    st.metric(

        "Confidence",

        f"{st.session_state.signal_confidence:.2%}",

    )

with signal_cols[2]:

    price = st.session_state.current_price

    st.metric(

        "Current Price",

        format_rupees(price) if price > 0 else "—",

    )

with signal_cols[3]:

    st.metric("Mode", active_mode())

# ============================================================

# MODEL PANEL

# ============================================================

st.subheader("Model Ensemble")

model_cols = st.columns(5)

for column, name in zip(

    model_cols,

    ["XGBoost", "LightGBM", "CatBoost", "LSTM", "Transformer"],

):

    with column:

        st.metric(name, "—", help="Live model probabilities will populate here.")

st.caption(

    "The unified terminal is ready for the existing five-model signal "

    "pipeline. Live model/data integration remains a separate execution step."

)

# ============================================================

# POSITION PANEL

# ============================================================

st.divider()

st.header(f"📌 {active_mode()} Position")

position = active_position()

price = active_current_price()

if active_mode() == "PAPER":

    quantity = st.session_state.paper_quantity

    entry = st.session_state.paper_entry_price

    unrealized = st.session_state.paper_unrealized_pnl

    charges = st.session_state.paper_charges

else:

    quantity = st.session_state.real_quantity

    entry = st.session_state.real_entry_price

    unrealized = 0.0

    charges = 0.0

position_cols = st.columns(5)

with position_cols[0]:

    st.metric("Position", position)

with position_cols[1]:

    st.metric("Quantity", f"{quantity:,}")

with position_cols[2]:

    st.metric(

        "Entry Price",

        format_rupees(entry) if entry > 0 else "—",

    )

with position_cols[3]:

    st.metric("Unrealized P&L", format_rupees(unrealized))

with position_cols[4]:

    st.metric("Charges", format_rupees(charges))

# ============================================================

# LIVE PAPER MARK-TO-MARKET

# ============================================================

def render_paper_live_panel() -> None:

    ltp = _feed_ltp()

    marks_live = st.session_state.paper_instrument == "NIFTY_FUTURES"

    has_positions = bool(st.session_state.paper_open_positions)

    if ltp > 0 and marks_live and (st.session_state.paper_running or has_positions):

        paper_mark_to_market(ltp)

        persist_to_shared_state()

    elif has_positions:

        # No live price for this profile: still enforce the session-close

        # rules at the last known price.

        apply_session_close_rules(st.session_state.paper_current_price)

        persist_to_shared_state()

    m = portfolio_metrics()

    phase = session_phase()

    sched = session_schedule()

    live_cols = st.columns(6)

    with live_cols[0]:

        st.metric("Open Positions", m["count"])

    with live_cols[1]:

        st.metric("Portfolio Exposure", format_rupees(m["exposure"]))

    with live_cols[2]:

        st.metric(

            "Portfolio Leverage",

            f"{m['leverage']:.2f}×",

            help="Total exposure of open positions ÷ capital. Closed trades excluded.",

        )

    with live_cols[3]:

        st.metric(

            "Avg Leverage (open trades)",

            f"{m['avg_leverage']:.2f}×",

            help="Average of each OPEN position's leverage. Closed trades excluded.",

        )

    with live_cols[4]:

        st.metric("Unrealized P&L", format_rupees(m["unrealized"]))

    with live_cols[5]:

        st.metric("Live Equity", format_rupees(st.session_state.paper_equity))

    if m["count"]:

        st.dataframe(

            [

                {

                    "Trade": p["trade_id"][:8],

                    "Direction": p["direction"],

                    "Qty": p["quantity"],

                    "Entry": p["entry_price"],

                    "Mark": _mark(p),

                    "Exposure": round(p["quantity"] * _mark(p), 2),

                    "Leverage": round(

                        p["quantity"] * _mark(p) / m["capital"], 2

                    ) if m["capital"] > 0 else 0.0,

                    "Unrealized": round(float(p.get("unrealized", 0.0)), 2),

                    "Since": p["entry_time"][11:19],

                }

                for p in _open_positions()

            ],

            use_container_width=True,

            hide_index=True,

        )

    phase_text = {

        "PRE_OPEN": f"Market not open — entries from {sched['market_open']:%H:%M}.",

        "OPEN": (

            f"Trading. Exit window opens {sched['exit_window']:%H:%M}; "

            f"all positions closed by {sched['forced_exit']:%H:%M}."

        ),

        "EXIT_WINDOW": (

            f"EXIT WINDOW — no new entries; positions exit at their best moment, "

            f"hard exit {sched['forced_exit']:%H:%M}."

        ),

        "FORCED_EXIT": (

            f"HARD EXIT — closing everything before broker square-off at "

            f"{sched['broker_square_off']:%H:%M}."

        ),

        "BROKER_SQUARE_OFF": (

            f"After {sched['broker_square_off']:%H:%M}: any open position is "

            f"auto-squared-off by the broker with a penalty of "

            f"{format_rupees(broker_penalty_per_position())} each."

        ),

    }[phase]

    if phase in ("EXIT_WINDOW", "FORCED_EXIT", "BROKER_SQUARE_OFF"):

        st.warning(phase_text)

    else:

        st.caption(phase_text)

    if not marks_live:

        st.caption(

            "Live marking uses the NIFTY index LTP, so it runs only for the "

            "NIFTY futures profile. Stock/option profiles need their own price feed."

        )

if active_mode() == "PAPER":

    st.subheader("Live Paper Mark-to-Market")

    if hasattr(st, "fragment"):

        @st.fragment(run_every="2s")

        def _paper_live_fragment():

            render_paper_live_panel()

        _paper_live_fragment()

    else:

        render_paper_live_panel()

# ============================================================

# PAPER PERFORMANCE

# ============================================================

st.divider()

st.header("📈 Paper Performance")

performance_cols = st.columns(5)

with performance_cols[0]:

    st.metric(

        "Gross P&L",

        format_rupees(st.session_state.paper_gross_pnl),

    )

with performance_cols[1]:

    st.metric(

        "Charges",

        format_rupees(st.session_state.paper_charges),

    )

with performance_cols[2]:

    st.metric(

        "Net P&L",

        format_rupees(st.session_state.paper_net_pnl),

    )

with performance_cols[3]:

    st.metric(

        "Trades",

        st.session_state.paper_trade_count,

    )

with performance_cols[4]:

    st.metric(

        "Max Drawdown",

        format_rupees(st.session_state.paper_max_drawdown),

    )

stats_cols = st.columns(4)

with stats_cols[0]:

    st.metric(

        "Winning Trades",

        st.session_state.paper_winning_trades,

    )

with stats_cols[1]:

    st.metric(

        "Losing Trades",

        st.session_state.paper_losing_trades,

    )

with stats_cols[2]:

    trade_count = st.session_state.paper_trade_count

    win_rate = (

        st.session_state.paper_winning_trades / trade_count

        if trade_count > 0

        else 0.0

    )

    st.metric("Win Rate", f"{win_rate:.2%}")

with stats_cols[3]:

    st.metric(

        "Paper Equity",

        format_rupees(st.session_state.paper_equity),

    )

st.caption(

    "Net P&L = Gross − Charges − Slippage − Penalties. Slippage so far: "

    + format_rupees(st.session_state.paper_slippage)

    + " · Auto square-off penalties: "

    + format_rupees(st.session_state.paper_penalties)

)

# ============================================================

# REAL ACCOUNT PANEL

# ============================================================

st.divider()

st.header("💰 Real Account")

if active_mode() == "REAL":

    real_cols = st.columns(4)

    with real_cols[0]:

        st.metric("Broker", BROKER_NAME)

    with real_cols[1]:

        st.metric("Execution", "LOCKED")

    with real_cols[2]:

        st.metric("Position", st.session_state.real_position)

    with real_cols[3]:

        st.metric("Real P&L", format_rupees(st.session_state.real_pnl))

    st.info(

        "Broker account/order/position synchronization will be connected "

        "before real execution is enabled."

    )

else:

    st.caption(

        "Switch to REAL mode to view the real-account control surface."

    )

# ============================================================

# PAPER TRADE JOURNAL

# ============================================================

st.divider()

st.header("📓 Paper Trade Journal")

if st.session_state.paper_trade_log:

    st.dataframe(

        st.session_state.paper_trade_log,

        use_container_width=True,

        hide_index=True,

    )

else:

    st.info("No paper trades yet.")

# ============================================================

# DAILY PAPER REPORT (generated from the ledger)

# ============================================================

st.divider()

st.header("📑 Daily Paper Report")

st.caption(

    "Generated from the append-only trade/event ledger in "

    f"{LEDGER_DIR}. The ledger is the source of truth; the report is "

    "always rebuilt from it."

)

_report_cols = st.columns([2, 1])

with _report_cols[0]:

    _report_day = st.date_input("Report date", value=datetime.now().date())

with _report_cols[1]:

    st.write("")

    _generate = st.button("📊 GENERATE DAILY REPORT", use_container_width=True)

if _generate:

    try:

        st.session_state["daily_report"] = build_daily_report(

            LEDGER,

            "PAPER",

            _report_day,

            float(st.session_state.paper_capital),

            MODEL_NAMES,

            paper_settings()["max_leverage"],

        )

    except Exception as exc:

        st.error(f"Report generation failed: {exc}")

_report = st.session_state.get("daily_report")

if _report:

    _concl = _report["conclusion"]

    _headline = f"Daily Result: {_concl['result']} — {_report['meta']['date']}"

    if _concl["result"] == "PROFITABLE":

        st.success(_headline)

    elif _concl["result"] == "LOSS":

        st.error(_headline)

    else:

        st.info(_headline)

    st.markdown("\n".join(f"- {line}" for line in _concl["lines"]))

    st.markdown("**Assessment**")

    st.markdown("\n".join(f"- {line}" for line in _concl["assessment"]))

    _tabs = st.tabs([

        "Capital & Exposure",

        "Trades",

        "Exposure Timeline",

        "P&L & Stats",

        "Models",

        "Risk",

    ])

    with _tabs[0]:

        _rows = []

        for _k, _v in _report["capital_exposure"].items():

            if "leverage" in _k.lower():

                _rows.append({"Metric": _k, "Value": f"{_v:.2f}×"})

            elif "%" in _k:

                _rows.append({"Metric": _k, "Value": f"{_v:.1f}%"})

            else:

                _rows.append({"Metric": _k, "Value": format_rupees(_v)})

        st.dataframe(_rows, use_container_width=True, hide_index=True)

    with _tabs[1]:

        if _report["trades"]:

            st.dataframe(_report["trades"], use_container_width=True, hide_index=True)

        else:

            st.info("No closed trades on this date.")

    with _tabs[2]:

        if _report["timeline"]:

            st.dataframe(_report["timeline"], use_container_width=True, hide_index=True)

        else:

            st.info("No exposure recorded on this date.")

    with _tabs[3]:

        _p = _report["pnl"]

        st.dataframe(

            [

                {"Item": "Gross trading P&L", "Amount": format_rupees(_p["gross_pnl"])},

                {"Item": "Brokerage", "Amount": format_rupees(-_p["brokerage"])},

                {"Item": "Taxes/fees", "Amount": format_rupees(-_p["taxes_fees"])},

                {"Item": "Slippage", "Amount": format_rupees(-_p["slippage"])},

                {"Item": "Auto square-off penalties", "Amount": format_rupees(-_p.get("penalties", 0.0))},

                {"Item": "Other costs", "Amount": format_rupees(-_p["other_costs"])},

                {"Item": "Net P&L", "Amount": format_rupees(_p["net_pnl"])},

            ],

            use_container_width=True,

            hide_index=True,

        )

        _s = _report["stats"]

        _pf = "∞" if _s["profit_factor"] == float("inf") else f"{_s['profit_factor']:.2f}"

        st.dataframe(

            [

                {"Metric": "Trades", "Value": str(_s["trades"])},

                {"Metric": "Winning / Losing", "Value": f"{_s['winning_trades']} / {_s['losing_trades']}"},

                {"Metric": "Win rate", "Value": f"{_s['win_rate']:.1%}"},

                {"Metric": "Profit factor", "Value": _pf},

                {"Metric": "Average winner", "Value": format_rupees(_s["avg_winner"])},

                {"Metric": "Average loser", "Value": format_rupees(_s["avg_loser"])},

                {"Metric": "Largest winner", "Value": format_rupees(_s["largest_winner"])},

                {"Metric": "Largest loser", "Value": format_rupees(_s["largest_loser"])},

                {"Metric": "Long P&L", "Value": format_rupees(_s["long_pnl"])},

                {"Metric": "Short P&L", "Value": format_rupees(_s["short_pnl"])},

                {"Metric": "Avg holding (min)", "Value": f"{_s['avg_holding_minutes']:.1f}"},

            ],

            use_container_width=True,

            hide_index=True,

        )

    with _tabs[4]:

        _m = _report["models"]

        st.dataframe(

            [{"Model": _n, **_v} for _n, _v in _m["per_model"].items()],

            use_container_width=True,

            hide_index=True,

        )

        st.write(

            f"Ensemble signals: {_m['signals']} "

            f"(directional {_m['directional_signals']}, neutral {_m['neutral_signals']}) · "

            f"Agreement rate: {_m['agreement_rate']:.1%} · "

            f"Trades generated: {_m['trades_generated']} · "

            f"Rejected: {_m['trades_rejected']}"

        )

        st.write("Confidence distribution:", _m["confidence_distribution"] or "—")

        st.write("P&L by confidence bucket:", _m["pnl_by_confidence"] or "—")

        st.write("P&L by model agreement:", _m["pnl_by_agreement"] or "—")

    with _tabs[5]:

        _r = _report["risk"]

        st.dataframe(

            [

                {"Metric": "Max intraday drawdown", "Value": format_rupees(_r["max_intraday_drawdown"])},

                {"Metric": "Max leverage / allowed", "Value": f"{_r['max_leverage']:.2f}× / {_r['allowed_leverage']:.2f}×"},

                {"Metric": "Max capital utilisation", "Value": f"{_r['max_capital_utilization_pct']:.1f}%"},

                {"Metric": "Max position size", "Value": str(_r["max_position_size"])},

                {"Metric": "Max simultaneous positions", "Value": str(_r["max_simultaneous_positions"])},

                {"Metric": "Max consecutive losses", "Value": str(_r["max_consecutive_losses"])},

                {"Metric": "Longest flat period (min)", "Value": f"{_r['longest_flat_minutes']:.1f}"},

                {"Metric": "Largest adverse excursion", "Value": format_rupees(_r["largest_adverse_excursion"])},

                {"Metric": "Largest favourable excursion", "Value": format_rupees(_r["largest_favorable_excursion"])},

                {"Metric": "Kill-switch events", "Value": str(_r["kill_switch_events"])},

                {"Metric": "Risk-rule violations", "Value": str(_r["risk_violations"])},

                {"Metric": "Rejected trades", "Value": str(_r["rejected_trades"])},

            ],

            use_container_width=True,

            hide_index=True,

        )

    _dl_cols = st.columns(2)

    with _dl_cols[0]:

        st.download_button(

            "⬇️ Download report (Markdown)",

            data=report_to_markdown(_report),

            file_name=f"athena_paper_report_{_report['meta']['date']}.md",

            mime="text/markdown",

            use_container_width=True,

        )

    with _dl_cols[1]:

        st.download_button(

            "⬇️ Download report (JSON)",

            data=json.dumps(_report, indent=2, default=str),

            file_name=f"athena_paper_report_{_report['meta']['date']}.json",

            mime="application/json",

            use_container_width=True,

        )

# ============================================================

# REAL TRADE JOURNAL

# ============================================================

st.header("📓 Real Trade Journal")

if st.session_state.real_trade_log:

    st.dataframe(

        st.session_state.real_trade_log,

        use_container_width=True,

        hide_index=True,

    )

else:

    st.info(

        "No real trades recorded. Real execution is locked in this build."

    )

# ============================================================

# ARCHITECTURE

# ============================================================

st.divider()

st.header("🏗️ Execution Architecture")

architecture_cols = st.columns(3)

with architecture_cols[0]:

    st.write("### Signal Engine")

    st.write(

        "Shared by PAPER and REAL modes. "

        "The same model predictions and risk decisions are intended "

        "to feed both execution paths."

    )

with architecture_cols[1]:

    st.write("### Paper Executor")

    st.write(

        "Simulated fills, simulated costs, independent paper P&L "

        "and a separate paper journal."

    )

with architecture_cols[2]:

    st.write("### Shoonya Executor")

    st.write(

        "Architecturally reserved for real execution. "

        "Order submission is locked until the forward-test and "

        "real-execution safety stages are completed."

    )

# ============================================================

# SAFETY

# ============================================================

st.divider()

st.header("🛡️ Safety")

safety_cols = st.columns(5)

with safety_cols[0]:

    st.write("**Current Mode**")

    st.write(active_mode())

with safety_cols[1]:

    st.write("**Real Orders**")

    st.write("DISABLED" if not REAL_ORDER_EXECUTION_ENABLED else "ENABLED")

with safety_cols[2]:

    st.write("**Shoonya API**")

    st.write("AUTH / SESSION")

with safety_cols[3]:

    st.write("**Paper Account**")

    st.write("SEPARATE")

with safety_cols[4]:

    st.write("**Kill Switch**")

    st.write("GLOBAL")

st.caption(

    "Athena 3.0 unified terminal: PAPER and REAL are separate execution "

    "paths. Real-money order submission is intentionally locked in V3."

)

# Persist shared process state so all browser sessions see the same Athena terminal.

persist_to_shared_state()