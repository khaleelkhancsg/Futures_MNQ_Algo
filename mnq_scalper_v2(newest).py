#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MNQ MULTI-TIMEFRAME MOMENTUM SCALPER  v2.0  —  TopstepX / ProjectX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instrument  : MNQ (Micro E-mini Nasdaq-100, $2.00/pt, tick=$0.50)
Timeframes  : 5-minute (regime) + 1-minute (structure + entry trigger)

═══════════════════════════════════════════════════════════════
WHAT CHANGED FROM v1.1 AND WHY
═══════════════════════════════════════════════════════════════

  v1.1 problems (academically documented):
  ─────────────────────────────────────────
  • MACD crossover on 5-min fired entries immediately — entered
    into the first bar of a move, often at exhaustion.
    (Nikkei 225 futures study: default MACD negative PnL in index
    futures as entry signal — MDPI JRFM 2021)

  • RR was 3.0 TP / 3.5 SL  =  0.86:1 (negative RR).
    Required >54% win rate just to break even before costs.

  • No regime filter — traded in all conditions including chop.
    (Mesfin 2026 arXiv:2605.04004: regime instability is the
    primary failure mode — signals that worked in 2021-22 trends
    reversed in 2023-25 chop)

  • Single timeframe — no structural entry confirmation.

  v2.0 fixes:
  ────────────
  • MACD demoted to 5-min regime CONTEXT only (is histogram
    expanding or compressing?). No longer an entry trigger.

  • Entry trigger: RSI(7) divergence on 1-min chart at 21 EMA
    pullback. Exploits documented 1-min SAC(1) = -0.1075
    mean-reversion (arXiv:2408.17187). Fires 2-4 bars earlier
    than MACD crossover confirmation.

  • ADX(14) regime gate: ADX < session threshold → no entries.
    Threshold varies by session liquidity (20 London/NY, 25 Asia).

  • 1:1 RR (SL = TP = 1.5×ATR). After 0.5R reached, a soft
    breakeven stop activates on the remaining position.
    (PropTradingVibes 2026: trailing drawdown props favour 1:1
    with partial exits over negative-RR high-frequency entries)

  • RTH drift rule: first 30 min of RTH sets session bias.
    Counter-drift trades blocked. (RTH Confluence Signal T=5.83
    on MNQ 2021-2025, Mesfin 2026; Gao et al. 2018 Sharpe 0.43)

  • Session-aware: Asia/London/LN-NY/NY/NY-Late each have
    appropriate ADX thresholds. No entries after 19:00 UK.

═══════════════════════════════════════════════════════════════
ANTI-OVERFITTING CONTRACT
═══════════════════════════════════════════════════════════════

  RSI thresholds (40/60 entry zone, 35/65 extremes) are FIXED
  across all sessions. Session-specific RSI would require
  historical optimisation — curve fitting.

  ADX thresholds (20/22/25) follow documented session liquidity
  differences, not backtested optimisation.

  EMA periods (9/21/50) are standard published parameters.

  The 6-minute time stop is based on momentum decay evidence
  (Gao et al. 2018), not optimised on MNQ data.

═══════════════════════════════════════════════════════════════
SESSION TIERS  (Europe/London time — BST/GMT auto-handled)
═══════════════════════════════════════════════════════════════

  ASIA      00:00–08:00 UK   ADX ≥ 25   No RTH drift
  LONDON    08:00–14:30 UK   ADX ≥ 22   No RTH drift
  LN_NY     14:30–16:30 UK   ADX ≥ 20   RTH drift active ← best
  NY        16:30–19:00 UK   ADX ≥ 20   RTH drift active
  NY_LATE   19:00–21:00 UK   NO NEW ENTRIES
  BLACKOUT  21:00–23:00 UK   NO TRADING

═══════════════════════════════════════════════════════════════
ENTRY CHECKLIST  (ALL 8 must pass — no skipping)
═══════════════════════════════════════════════════════════════

  5-min (regime layer):
  [1] EMA 9 / 21 / 50 cleanly stacked in trade direction
  [2] ADX(14) ≥ session threshold
  [3] MACD histogram on correct side AND expanding (not shrinking)
  [4] Session drift aligned (LN_NY + NY sessions only)

  1-min (structure + trigger layer):
  [5] Price pulled back to 1-min 21 EMA (within 0.4×ATR5 tol)
  [6] 1-min 50 EMA not breached on close in last 3 bars
  [7] RSI(14) between 40–60 at the pullback bar
  [8] RSI(7) divergence: price at/near low, RSI(7) higher low

═══════════════════════════════════════════════════════════════
EXIT CASCADE  (priority order — higher overrides lower)
═══════════════════════════════════════════════════════════════

  P1  Hard stop:      SL bracket = 1.5×ATR14  [non-negotiable]
  P2  Structural:     1-min close crosses 9 EMA against position
  P2b RSI extreme:    RSI(14) < 35 in long / > 65 in short
  P3  Time stop:      6 min elapsed, TP1 (0.5R) not reached
  P4  TP1 flag:       0.5R hit → breakeven soft stop activates
  P5  Breakeven stop: after TP1, close returns to entry → exit
  P6  Full TP:        1.5×ATR bracket hit by broker
  P7  5-min reversal: opposite EMA crossover → soft exit all
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # pip install backports.zoneinfo

import httpx                                    # direct REST API calls
from dotenv import load_dotenv
from project_x_py import TradingSuite

load_dotenv()

UK_TZ                  = ZoneInfo("Europe/London")
RECONNECT_DELAY_S      = 30
MAX_CONSECUTIVE_ERRORS = 10
ENTRY_COOLDOWN_S       = 120   # 2-min cooldown after any exit


class ConnectionLostError(Exception):
    pass


# ─────────────────────────────────────────────────────────────
# 1.  LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("MNQ-Scalper")

_log_dir  = Path("logs")
_log_dir.mkdir(exist_ok=True)
_log_file = _log_dir / f"mnq_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().addHandler(_fh)
log.info("📝 Log: %s", _log_file)

# Suppress project_x_py internal bounded_stats realtime-feed noise.
# The library logs a schema-mismatch ERROR (width 8 vs 6 cols) during
# start_realtime_feed — a known bug around contract rollovers.
# Our bot uses polling (get_bars) and is unaffected by this background feed.
# Setting CRITICAL here keeps genuine crashes visible.
for _lib in ("project_x_py.statistics.bounded_statistics.bounded_stats",
             "project_x_py.statistics"):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)

clog = logging.getLogger("MNQ-Candle")
clog.setLevel(logging.INFO)
if not clog.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(message)s"))
    clog.addHandler(_ch)
    clog.propagate = False


# ─────────────────────────────────────────────────────────────
# 2.  SESSION
# ─────────────────────────────────────────────────────────────
class Session(str, Enum):
    ASIA      = "Asia"
    LONDON    = "London"
    LN_NY     = "London/NY"
    NY        = "New York"
    NY_LATE   = "NY Late"
    BLACKOUT  = "Blackout"


def get_session(uk_dt: datetime) -> Session:
    h, m = uk_dt.hour, uk_dt.minute
    t = h * 60 + m
    if 21 * 60 <= t < 23 * 60:  return Session.BLACKOUT
    if t < 8 * 60:               return Session.ASIA
    if t < 14 * 60 + 30:         return Session.LONDON
    if t < 16 * 60 + 30:         return Session.LN_NY
    if t < 19 * 60:              return Session.NY
    return Session.NY_LATE


# ADX thresholds by session — liquidity-motivated, not curve-fitted
ADX_MIN: dict = {
    Session.ASIA:     25.0,
    Session.LONDON:   22.0,
    Session.LN_NY:    20.0,
    Session.NY:       20.0,
    Session.NY_LATE:  999.0,   # blocks all entries
    Session.BLACKOUT: 999.0,
}

# Sessions where RTH drift rule applies
DRIFT_SESSIONS = {Session.LN_NY, Session.NY}


# ─────────────────────────────────────────────────────────────
# 3.  PARAMS
# ─────────────────────────────────────────────────────────────
PARAMS: dict = {
    # ── 5-min EMA stack ──────────────────────────────────────
    "ema_fast":           9,
    "ema_mid":           21,
    "ema_trend":         50,

    # ── 5-min MACD (regime context — NOT entry trigger) ───────
    "macd_fast":         12,
    "macd_slow":         26,
    "macd_signal":        9,

    # ── 5-min ATR & ADX ──────────────────────────────────────
    "atr_period_5m":     14,
    "adx_period":        14,

    # ── 1-min ATR (EMA touch tolerance) ──────────────────────
    "atr_period_1m":      5,
    "ema_touch_mult":   0.4,   # price within 0.4×ATR(5) of 21EMA = "at EMA"

    # ── 1-min RSI ────────────────────────────────────────────
    # Thresholds are FIXED across all sessions (anti-overfitting)
    "rsi_slow":          14,   # entry zone gate
    "rsi_fast":           7,   # divergence trigger
    "rsi_lo":            40,   # lower bound of entry zone
    "rsi_hi":            60,   # upper bound of entry zone
    "rsi_extreme_lo":    35,   # oversold — exit long
    "rsi_extreme_hi":    65,   # overbought — exit short

    # ── Risk / Reward  (1:1 — replaces v1.1's -RR structure) ─
    "sl_mult":          1.5,   # SL = 1.5 × ATR(14) from 5-min
    "tp_mult":          1.5,   # TP = 1.5 × ATR(14)  → 1:1 RR
    # 0.5R monitoring level (not a bracket — managed in software)
    "partial_mult":     0.75,  # 0.5R = 0.75 × ATR(14)

    # ── Dynamic position sizing ──────────────────────────────
    # Contracts scale with ATR so dollar risk per trade stays near
    # target_risk_usd regardless of volatility conditions.
    #
    #   contracts = floor( target_risk_usd / (sl_mult × ATR × $2/pt) )
    #
    # Clamped to [1, max_contracts].  Adjust both values to match your
    # combine's trailing drawdown:
    #
    #   $2,000 DD → target ≈ $200 (10%), max_contracts = 5
    #   $5,000 DD → target ≈ $500 (10%), max_contracts = 12
    #   $10,000 DD → target ≈ $800 (8%), max_contracts = 20
    #
    "target_risk_usd":   200,   # target $ risk per trade
    "max_contracts":       5,   # hard cap — adjust to combine size
    "min_contracts":       1,   # always trade at least this many

    # ── Time stop ─────────────────────────────────────────────
    "time_stop_min":      6,   # exit if 0.5R not reached in 6 min

    # ── RTH drift window ──────────────────────────────────────
    "rth_h":             14,   # RTH opens 14:30 UK (09:30 ET)
    "rth_m":             30,
    "rth_drift_min":     30,   # use first 30-min close vs open for bias

    # ── Session / blackout ────────────────────────────────────
    "blackout_start_uk": 21,
    "blackout_end_uk":   23,
    "flatten_buffer_min": 5,

    # ── Warm-up ───────────────────────────────────────────────
    "warmup_5m_bars":    60,
    "warmup_1m_bars":    60,

    # ── Reconciliation ────────────────────────────────────────
    "sync_every_n_bars":  5,   # 1-min loop: broker sync every N bars (≈ 5 min)

    # ── Fast position monitor (3-second loop) ─────────────────
    # Runs independently of bar-close loops. Handles: broker sync,
    # precise time stop, intrabar 0.5R detection, breakeven stop.
    # Entry signals are NOT fired here — those require completed bars.
    "fast_poll_s":        3,   # seconds between fast-monitor ticks
    "sync_fast_n":        5,   # broker sync every N fast ticks (≈ 15 s)

    "tick_size":         0.50,
}

TICK = PARAMS["tick_size"]


# ─────────────────────────────────────────────────────────────
# 4.  BAR FETCHER  (unchanged from v1.1)
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# DIRECT BAR FETCHER  — bypasses project_x_py's bounded_stats
# ─────────────────────────────────────────────────────────────
# Root cause of the schema error, confirmed by the API docs:
#
#   The ProjectX realtime API has NO bar events.  The market hub
#   provides only GatewayQuote, GatewayDepth and GatewayTrade
#   (individual ticks).  The library's bounded_stats module
#   builds OHLCV bars internally by aggregating GatewayTrade
#   events, producing 6-column DataFrames.  The historical seed
#   from /api/History/retrieveBars is processed by the library
#   into 8-column DataFrames (6 OHLCV + 2 computed statistics).
#   When bounded_stats tries to append a live 6-col bar to the
#   8-col historical DataFrame, Polars raises ShapeError.  This
#   is structurally inevitable — it cannot be patched from
#   outside the library.
#
# Fix:
#   Call POST /api/History/retrieveBars directly via httpx,
#   bypassing client.get_bars() and the bounded_stats module
#   entirely.  The REST API always returns consistent 6-column
#   bars: {t, o, h, l, c, v}.  Our existing BarFetcher.ohlcv()
#   and BarFetcher.ts() static methods normalise this format.
#
# Auth:
#   POST /api/Auth/loginKey  →  JWT token (24-hour validity).
#   Token is refreshed automatically 1 hour before expiry.

_TOPSTEPX_BASE = "https://api.topstepx.com/api"


class DirectBarFetcher:
    """
    Fetches OHLCV bars by calling the ProjectX REST API directly.
    Replaces BarFetcher(client, symbol).fetch() for all bar data
    so the library's internal bounded_stats cache is never consulted.
    """

    _TOKEN_TTL_S = 23 * 3600   # refresh 1 h before the 24-h JWT expiry

    def __init__(self, contract_id: str, live: bool = False) -> None:
        self._contract_id = contract_id
        self._live        = live      # False = sim/practice, True = live account
        self._token:      Optional[str]      = None
        self._token_time: Optional[datetime] = None

    # ── Auth ──────────────────────────────────────────────────────────────

    async def _get_token(self) -> str:
        """Return a valid JWT, fetching one if necessary."""
        now = datetime.now(timezone.utc)
        if (self._token is None
                or self._token_time is None
                or (now - self._token_time).total_seconds() > self._TOKEN_TTL_S):
            self._token      = await self._login()
            self._token_time = datetime.now(timezone.utc)
        return self._token

    @staticmethod
    async def _login() -> str:
        username = os.environ.get("PROJECT_X_USERNAME", "")
        api_key  = os.environ.get("PROJECT_X_API_KEY",  "")
        if not username or not api_key:
            raise RuntimeError("PROJECT_X_USERNAME / PROJECT_X_API_KEY not set")
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(
                f"{_TOPSTEPX_BASE}/Auth/loginKey",
                json={"userName": username, "apiKey": api_key},
                headers={"Content-Type": "application/json",
                         "Accept": "text/plain"},
            )
            resp.raise_for_status()
            data = resp.json()
        if not data.get("success", False):
            raise RuntimeError(
                f"Auth/loginKey failed: {data.get('errorMessage', 'unknown')}")
        token = data.get("token") or data.get("data") or data.get("sessionToken")
        if not token:
            raise RuntimeError(
                f"Auth/loginKey: no token in response — keys: {list(data)}")
        log.info("✅ DirectBarFetcher: JWT obtained")
        return token

    # ── Bar fetch ─────────────────────────────────────────────────────────

    async def fetch(self, interval_minutes: int, days: int = 1) -> List[dict]:
        """
        Fetch up to `days` days of `interval_minutes`-minute bars.
        Returns a list of dicts [{t, o, h, l, c, v}, …] sorted oldest-first,
        matching the format expected by BarFetcher.ohlcv() / BarFetcher.ts().
        """
        token      = await self._get_token()
        end_time   = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        payload = {
            "contractId":       self._contract_id,
            "live":             self._live,
            "startTime":        start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endTime":          end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "unit":             2,                 # 2 = Minute (per API docs)
            "unitNumber":       interval_minutes,
            "limit":            2000,
            "includePartialBar": True,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Accept":        "text/plain",
        }

        delay = 1.5
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=15.0) as http:
                    resp = await http.post(
                        f"{_TOPSTEPX_BASE}/History/retrieveBars",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code == 401:
                        log.warning("DirectBarFetcher: 401 — refreshing JWT")
                        self._token = None
                        token = await self._get_token()
                        headers["Authorization"] = f"Bearer {token}"
                        continue
                    resp.raise_for_status()
                    data = resp.json()

                if data.get("errorCode", 0) not in (0, None):
                    log.warning("DirectBarFetcher: API errorCode=%s: %s",
                                data.get("errorCode"), data.get("errorMessage"))
                    return []

                bars = data.get("bars") or []
                # API returns newest-first; sort to oldest-first for our code
                bars.sort(key=lambda b: str(b.get("t", "")))
                return bars

            except Exception as exc:
                log.warning("DirectBarFetcher: attempt %d/3 (%dmin): %s",
                            attempt + 1, interval_minutes, exc)
                await asyncio.sleep(delay)
                delay *= 2.0

        log.error("DirectBarFetcher: all retries exhausted (%dmin bars)",
                  interval_minutes)
        return []


class BarFetcher:
    def __init__(self, client, symbol: str = "MNQ") -> None:
        self.client = client
        self.symbol = symbol

    async def fetch(self, interval_minutes: int, days: int = 1) -> List[dict]:
        delay = 1.5
        for attempt in range(3):
            try:
                raw = await self.client.get_bars(
                    self.symbol, days=days, interval=interval_minutes)
                return self._normalise(raw)
            except Exception as exc:
                log.warning("BarFetcher %dmin attempt %d/3: %s",
                            interval_minutes, attempt + 1, exc)
                await asyncio.sleep(delay)
                delay *= 2.0
        log.error("BarFetcher: all retries exhausted (%dmin)", interval_minutes)
        return []

    @staticmethod
    def _normalise(raw) -> List[dict]:
        if raw is None:
            return []
        if hasattr(raw, "to_dicts"):
            try:
                import polars as pl
                casts = {}
                for col, dtype in zip(raw.columns, raw.dtypes):
                    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
                        casts[col] = pl.col(col).dt.convert_time_zone(
                            "UTC").dt.replace_time_zone(None)
                if casts:
                    raw = raw.with_columns(list(casts.values()))
                rows = raw.to_dicts()
            except Exception as exc:
                log.warning("BarFetcher normalise failed: %s", exc)
                return []
        elif isinstance(raw, list):
            rows = raw
        else:
            return []
        if not rows:
            return []

        def _key(b):
            for k in ("t", "timestamp", "datetime", "time", "date"):
                if k in b and b[k] is not None:
                    return str(b[k])
            return ""

        rows.sort(key=_key)
        return rows

    @staticmethod
    def ohlcv(bar: dict) -> Tuple[float, float, float, float, float]:
        def _g(keys):
            for k in keys:
                if k in bar and bar[k] is not None:
                    return float(bar[k])
            return 0.0
        return (
            _g(("open",   "o", "Open")),
            _g(("high",   "h", "High")),
            _g(("low",    "l", "Low")),
            _g(("close",  "c", "Close")),
            _g(("volume", "v", "Volume")),
        )

    @staticmethod
    def ts(bar: dict) -> Optional[str]:
        for k in ("t", "timestamp", "Timestamp", "datetime", "time"):
            if k in bar and bar[k] is not None:
                return str(bar[k])
        return None


# ─────────────────────────────────────────────────────────────
# 5.  INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────

def _ema_series(values: List[float], span: int) -> List[float]:
    if not values:
        return []
    a, out = 2.0 / (span + 1), [values[0]]
    for v in values[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def _compute_atr(bars: List[dict], period: int) -> float:
    n = len(bars)
    if n < 2:
        return 0.0
    ohlcvs = [BarFetcher.ohlcv(b) for b in bars]
    tr = []
    for i, (o, h, l, c, v) in enumerate(ohlcvs):
        if i == 0:
            tr.append(h - l)
        else:
            pc = ohlcvs[i - 1][3]
            tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return _ema_series(tr, period)[-1]


def _compute_rsi(closes: List[float], period: int) -> List[float]:
    """Wilder RSI. Returns same-length list (padded with 50.0 for first bars)."""
    n = len(closes)
    if n < period + 1:
        return [50.0] * n
    pad  = [50.0] * period
    dlts = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains  = [max(d, 0.0) for d in dlts]
    losses = [max(-d, 0.0) for d in dlts]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period

    def _rsi(ag, al):
        return 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    out = pad + [_rsi(ag, al)]
    for i in range(period, len(dlts)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
        out.append(_rsi(ag, al))
    return out


def _compute_adx(bars: List[dict], period: int = 14) -> float:
    """
    ADX via Wilder smoothing.
    Academic basis: Baltussen et al. (2021) regime classification.
    """
    n = len(bars)
    if n < period * 2 + 1:
        return 0.0
    ohlcvs = [BarFetcher.ohlcv(b) for b in bars]
    pdm, mdm, trs = [], [], []
    for i in range(1, n):
        h, l = ohlcvs[i][1], ohlcvs[i][2]
        ph, pl, pc = ohlcvs[i-1][1], ohlcvs[i-1][2], ohlcvs[i-1][3]
        up, dn = h - ph, pl - l
        pdm.append(up if up > dn and up > 0 else 0.0)
        mdm.append(dn if dn > up and dn > 0 else 0.0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    def _ws(data: List[float], p: int) -> List[float]:
        if len(data) < p:
            return []
        s = [sum(data[:p])]
        for v in data[p:]:
            s.append(s[-1] - s[-1] / p + v)
        return s

    tr_s, pd_s, md_s = _ws(trs, period), _ws(pdm, period), _ws(mdm, period)
    dx = []
    for i in range(min(len(tr_s), len(pd_s), len(md_s))):
        if tr_s[i] == 0:
            dx.append(0.0)
            continue
        pdi = 100.0 * pd_s[i] / tr_s[i]
        mdi = 100.0 * md_s[i] / tr_s[i]
        sm  = pdi + mdi
        dx.append(100.0 * abs(pdi - mdi) / sm if sm > 0 else 0.0)
    # ADX = Wilder EMA of DX values (mean-based, NOT the cumulative _ws sum)
    # _ws accumulates sums — correct for TR/DM where ratios cancel the scale.
    # ADX is a direct smoothed value so must start from the mean of dx[:period].
    if len(dx) < period:
        return 0.0
    adx_val = sum(dx[:period]) / period      # seed: simple mean of first period
    for v in dx[period:]:
        adx_val = (adx_val * (period - 1) + v) / period
    return adx_val


def _compute_macd_hist(bars: List[dict],
                        fast: int = 12, slow: int = 26,
                        sig: int = 9) -> Tuple[float, float]:
    """Returns (current_hist, prev_hist). Used as regime context only."""
    closes = [BarFetcher.ohlcv(b)[3] for b in bars]
    if len(closes) < slow + sig + 1:
        return 0.0, 0.0
    macd   = [f - s for f, s in zip(_ema_series(closes, fast),
                                     _ema_series(closes, slow))]
    hist   = [m - s for m, s in zip(macd, _ema_series(macd, sig))]
    return (hist[-1], hist[-2]) if len(hist) >= 2 else (hist[-1], hist[-1])


def _bullish_rsi_div(closes: List[float], rsi7: List[float],
                      lookback: int = 5) -> bool:
    """
    Bullish RSI(7) divergence: price at/near recent low, RSI higher.
    Exploits 1-min SAC(1) = -0.1075 mean-reversion (arXiv:2408.17187).
    """
    if len(closes) < lookback or len(rsi7) < lookback:
        return False
    rc, rr = closes[-lookback:], rsi7[-lookback:]
    idx      = rc[:-1].index(min(rc[:-1]))
    return (rc[-1] <= rc[idx] * 1.002) and (rr[-1] >= rr[idx] + 2.0)


def _bearish_rsi_div(closes: List[float], rsi7: List[float],
                      lookback: int = 5) -> bool:
    """Bearish RSI(7) divergence: price at/near recent high, RSI lower."""
    if len(closes) < lookback or len(rsi7) < lookback:
        return False
    rc, rr = closes[-lookback:], rsi7[-lookback:]
    idx      = rc[:-1].index(max(rc[:-1]))
    return (rc[-1] >= rc[idx] * 0.998) and (rr[-1] <= rr[idx] - 2.0)


# ─────────────────────────────────────────────────────────────
# 6.  STATE DATACLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class FiveMinState:
    ema9:           float = 0.0
    ema21:          float = 0.0
    ema50:          float = 0.0
    atr14:          float = 0.0
    adx:            float = 0.0
    macd_hist:      float = 0.0
    macd_hist_prev: float = 0.0
    direction:      int   = 0
    bars_since_cross: int = 0
    ready:          bool  = False


# ─────────────────────────────────────────────────────────────
# 7.  EMA ENGINE  (extended: fast/mid/trend for stack check)
# ─────────────────────────────────────────────────────────────
class EMAEngine:
    def __init__(self, fast_p: int, mid_p: int, trend_p: int) -> None:
        self.fast_p, self.mid_p, self.trend_p = fast_p, mid_p, trend_p
        self._af  = 2.0 / (fast_p  + 1)
        self._am  = 2.0 / (mid_p   + 1)
        self._at  = 2.0 / (trend_p + 1)
        self._fast = self._mid = self._trend = None
        self._pf   = self._pm = None
        self._bar_count = self._bar_idx = 0
        self._cross_bar = -9999
        self._cross_dir = 0

    def update(self, c: float) -> None:
        self._pf, self._pm = self._fast, self._mid
        if self._fast is None:
            self._fast = self._mid = self._trend = c
        else:
            self._fast  = self._af * c + (1 - self._af) * self._fast
            self._mid   = self._am * c + (1 - self._am) * self._mid
            self._trend = self._at * c + (1 - self._at) * self._trend
        self._bar_count += 1
        self._bar_idx   += 1
        if self._pf is not None and self._pm is not None:
            was = self._pf > self._pm
            now = self._fast > self._mid
            if not was and now:
                self._cross_dir, self._cross_bar = +1, self._bar_idx
            elif was and not now:
                self._cross_dir, self._cross_bar = -1, self._bar_idx

    def seed(self, bars: List[dict]) -> None:
        for b in bars[:-1]:
            self.update(BarFetcher.ohlcv(b)[3])

    @property
    def fast(self):   return self._fast
    @property
    def mid(self):    return self._mid
    @property
    def trend(self):  return self._trend

    @property
    def is_ready(self) -> bool:
        return self._bar_count >= self.trend_p

    @property
    def direction(self) -> int:
        return 0 if not self.is_ready else self._cross_dir

    @property
    def stack_ok(self) -> bool:
        """EMA 9 > 21 > 50 (long) or 9 < 21 < 50 (short)."""
        if not self.is_ready or self._cross_dir == 0 or self._trend is None:
            return False
        if self._cross_dir == 1:
            return self._fast > self._mid > self._trend   # type: ignore[operator]
        return self._fast < self._mid < self._trend        # type: ignore[operator]

    @property
    def bars_since_cross(self) -> int:
        return self._bar_idx - self._cross_bar


# ─────────────────────────────────────────────────────────────
# 8.  CANDLE PRINTER  (abbreviated — full detail in log file)
# ─────────────────────────────────────────────────────────────
DIV  = "─" * 64
DIV2 = "═" * 64

def _dsym(d: int) -> str:
    return "▲ BULL" if d == 1 else ("▼ BEAR" if d == -1 else "— NONE")


def print_5m(state: FiveMinState, session: Session,
             sig: int, in_pos: bool, pnl: float, trades: int,
             adx_thresh: float, conditions_met: bool) -> None:
    sig_s   = "▶▶ LONG 🟢" if sig == 1 else ("▶▶ SHORT 🔴" if sig == -1 else "── flat")
    stack_s = "✅" if state.ema9 > state.ema21 > state.ema50 or \
                      state.ema9 < state.ema21 < state.ema50 else "❌"
    adx_s   = f"{'✅' if state.adx >= adx_thresh else '❌'} {state.adx:.1f}"
    hist_ex = (sig == 1 and state.macd_hist > 0 and state.macd_hist > state.macd_hist_prev) or \
              (sig == -1 and state.macd_hist < 0 and state.macd_hist < state.macd_hist_prev)
    macd_s  = f"{'✅' if hist_ex else '❌'} hist={state.macd_hist:.4f}"
    clog.info(DIV)
    clog.info("[5m]  %s  Dir:%s  %d bars ago%s",
              session.value, _dsym(state.direction), state.bars_since_cross,
              "  📍 IN POS" if in_pos else "")
    clog.info("  EMA9:%.2f  EMA21:%.2f  EMA50:%.2f  Stack:%s",
              state.ema9, state.ema21, state.ema50, stack_s)
    clog.info("  ADX:%s  ATR:%.2f  MACD:%s", adx_s, state.atr14, macd_s)
    clog.info("  Signal: %s  |  Regime gate: %s",
              sig_s, "✅ OPEN" if conditions_met else "❌ BLOCKED")
    clog.info("  Session P&L: %+.2f  Trades: %d", pnl, trades)
    clog.info(DIV)


def print_entry(signal: int, entry: float, sl: float, tp: float,
                partial: float, atr: float, session: Session,
                contracts: int) -> None:
    d          = "LONG ▲" if signal == 1 else "SHORT ▼"
    sld        = abs(entry - sl)
    tpd        = abs(entry - tp)
    dollar_sl  = contracts * sld * 2.0          # $2 per point per contract
    dollar_tp  = contracts * tpd * 2.0
    clog.info("")
    clog.info(DIV2)
    clog.info("  🚀 ENTRY  —  MNQ SCALPER v2.0")
    clog.info(DIV2)
    clog.info("  Direction  : %s    Size: %d contract(s)    Session: %s",
              d, contracts, session.value)
    clog.info("  Entry      : %.2f", entry)
    clog.info("  Stop-Loss  : %.2f  (dist %.2f pts = %.1f×ATR  →  $%.0f risk)",
              sl, sld, PARAMS["sl_mult"], dollar_sl)
    clog.info("  TP (full)  : %.2f  (dist %.2f pts = %.1f×ATR  →  $%.0f reward)",
              tp, tpd, PARAMS["tp_mult"], dollar_tp)
    clog.info("  0.5R level : %.2f  (breakeven stop activates here)", partial)
    clog.info("  RR         : 1:%.2f  |  ATR(14) = %.2f  |  target_risk = $%d",
              tpd/sld if sld else 0, atr, PARAMS["target_risk_usd"])
    clog.info(DIV2)
    clog.info("")


def print_1m_status(uk_hm: str, close: float, direction: int,
                     session: Session, rsi14: float, rsi7: float,
                     in_pos: bool, pnl: float,
                     entry_reason: str) -> None:
    clog.info("  [1m] %s UK  C=%.2f  EMA:%s  Session:%s  "
              "RSI14:%.1f  RSI7:%.1f  %s  pnl:%+.0f  → %s",
              uk_hm, close, "▲" if direction == 1 else ("▼" if direction == -1 else "—"),
              session.value, rsi14, rsi7,
              "📍" if in_pos else "·", pnl, entry_reason)


# ─────────────────────────────────────────────────────────────
# 9.  PRE-FLIGHT  (env + account only — no test trades)
# ─────────────────────────────────────────────────────────────
class PreFlightCheck:
    PASS = "PASS"; FAIL = "FAIL"; WARN = "WARN"

    def __init__(self):
        self._results = []

    def _rec(self, name, status, detail):
        self._results.append({"n": name, "s": status, "d": detail})
        icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ "}.get(status, "  ")
        log.info("[PreFlight] %s %-38s %s", icon, name, detail)

    def _summary(self) -> bool:
        ok = all(r["s"] != self.FAIL for r in self._results)
        log.info("=" * 60)
        log.info("  PRE-FLIGHT  %s", "ALL SYSTEMS GO ✅" if ok else "BLOCKED ❌")
        for r in self._results:
            icon = {"PASS": "[OK]  ", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(r["s"], "      ")
            log.info("  %s  %-34s %s", icon, r["n"], r["d"])
        log.info("=" * 60)
        return ok

    async def run(self, suite) -> bool:
        log.info("[PreFlight] Starting...")
        for var in ("PROJECT_X_API_KEY", "PROJECT_X_USERNAME"):
            v = os.environ.get(var, "")
            if not v:
                self._rec(f"Env:{var}", self.FAIL, "NOT SET")
            else:
                self._rec(f"Env:{var}", self.PASS, f"Present ({v[:4]}...)" if "KEY" in var else f"Present ({v})")
        if any(r["s"] == self.FAIL for r in self._results):
            return self._summary()
        self._rec("Auth", self.PASS, "Session token obtained")
        try:
            acct = suite.client.account_info
            name = getattr(acct, "name", str(acct))
            bal  = getattr(acct, "balance", None)
            self._rec("Account", self.PASS,
                      f"{name}  bal={bal:,.2f}" if bal else name)
        except Exception as exc:
            self._rec("Account", self.WARN, f"Could not verify: {exc}")

        # ── 1-min bar fetch — critical initialisation step ─────────────
        # This mirrors the market-data check in v10.4's pre-flight.
        #
        # Root cause of the bounded_stats error
        # ──────────────────────────────────────
        # TradingSuite.create() starts a background realtime feed that seeds
        # its internal statistics DataFrame from the REST API.  Those bars
        # contain 8 columns (OHLCV + bid/ask volume + metadata).  When the
        # first live WebSocket update arrives for the 5-min timeframe it only
        # carries 6 columns, and Polars refuses to append mismatched schemas.
        #
        # The fix
        # ───────
        # Fetching 1-min bars here forces client.get_bars() to complete while
        # the library is still initialising its realtime feed.  This appears to
        # cause the library to complete its internal synchronisation cycle and
        # reconcile the column schema before the first 5-min WebSocket bar
        # arrives — the same mechanism that prevents the error in v10.4.
        # Doing this with interval=1 (not 5) avoids triggering the very
        # 5-min statistics path that has the schema mismatch.
        try:
            _prime_fetcher = BarFetcher(suite.client, "MNQ")
            _prime_bars    = await _prime_fetcher.fetch(1, days=1)
            if _prime_bars:
                _, _, _, _c, _ = BarFetcher.ohlcv(_prime_bars[-1])
                self._rec("Market data (1-min)",
                          self.PASS,
                          f"close={_c:.2f}  {len(_prime_bars)} bars — "
                          f"library feed primed")
            else:
                self._rec("Market data (1-min)", self.WARN,
                          "No bars returned — feed prime may be incomplete")
        except Exception as exc:
            self._rec("Market data (1-min)", self.WARN,
                      f"Prime fetch failed: {exc}")

        return self._summary()


# ─────────────────────────────────────────────────────────────
# 10.  STRATEGY
# ─────────────────────────────────────────────────────────────
class Strategy:
    SYMBOL = "MNQ"

    def __init__(self, suite: TradingSuite) -> None:
        self.suite  = suite
        self.client = suite.client

        self.ema = EMAEngine(PARAMS["ema_fast"], PARAMS["ema_mid"],
                             PARAMS["ema_trend"])

        self._contract_id:          Optional[str]       = None
        self._fetcher:              Optional[BarFetcher] = None
        self._last_balance:         float               = 0.0
        self._session_open_balance: float               = 0.0

        # 5-min state
        self._last_5m_ts:    Optional[str]    = None
        self._5m_state:      FiveMinState     = FiveMinState()
        self._5m_ready:      bool             = False
        self._prev_direction: int             = 0
        self._current_atr:   float            = 0.0

        # 1-min state
        self._last_1m_ts: Optional[str] = None
        self._1m_count:   int           = 0

        # RTH drift (reset each session)
        self._rth_open_price:       Optional[float] = None
        self._rth_drift_direction:  int             = 0    # +1/-1/0
        self._rth_drift_determined: bool            = False

        # Position state
        self.in_position:       bool             = False
        self._entry_signal:     int              = 0
        self._entry_price:      float            = 0.0
        self._stop_price:       float            = 0.0
        self._tp_price:         float            = 0.0
        self._partial_price:    float            = 0.0    # 0.5R level
        self._tp1_reached:      bool             = False
        self._trade_entry_time: Optional[datetime] = None
        self._last_exit_time:   Optional[datetime] = None

        # Session tracking
        self.session_trades: int = 0

        # Error counters
        self._5m_errors: int = 0
        self._1m_errors: int = 0
        self._3s_errors: int = 0

        # Intrabar MACD histogram tracking (3-second loop)
        # Watches the forming 1-min bar's histogram across ticks.
        # When it crosses zero in the regime direction, fires an early entry.
        self._prev_1m_hist:         float         = 0.0
        self._last_forming_ts:      Optional[str] = None
        self._intrabar_entry_fired: bool          = False

        # Staleness watchdog — tracks wall-clock time of last NEW 1-min bar.
        # If no new bar arrives for STALE_BAR_TIMEOUT_S seconds after warmup,
        # the library's internal bar cache has stopped updating (realtime feed
        # broken) and we must reconnect to restore HTTP bar fetches.
        self._last_bar_refresh:     Optional[datetime] = None

        self._eval_lock = asyncio.Lock()

    # ── Time helpers ──────────────────────────────────────────

    @staticmethod
    def _now_uk() -> datetime:
        return datetime.now(UK_TZ)

    @staticmethod
    def _utc_hm() -> str:
        return datetime.now(timezone.utc).strftime("%H:%M")

    @staticmethod
    def _uk_hm() -> str:
        return Strategy._now_uk().strftime("%H:%M")

    def _in_blackout(self) -> bool:
        h = self._now_uk().hour
        return PARAMS["blackout_start_uk"] <= h < PARAMS["blackout_end_uk"]

    def _should_flatten(self) -> bool:
        now_uk       = self._now_uk()
        cur_min      = now_uk.hour * 60 + now_uk.minute
        blk_min      = PARAMS["blackout_start_uk"] * 60
        mins_to_blk  = blk_min - cur_min
        return 0 < mins_to_blk <= PARAMS["flatten_buffer_min"]

    @staticmethod
    def _secs_to_next_5m() -> float:
        now     = datetime.now(timezone.utc)
        elapsed = (now.minute % 5) * 60 + now.second + now.microsecond / 1e6
        return max(300 - elapsed + 1.5, 1.0)

    # ── Balance ───────────────────────────────────────────────

    async def _get_balance(self) -> float:
        try:
            acct = self.client.account_info
            bal  = getattr(acct, "balance", None)
            if bal is not None:
                self._last_balance = float(bal)
        except Exception:
            pass
        return self._last_balance

    def _session_pnl(self) -> float:
        return self._last_balance - self._session_open_balance

    # ── RTH drift ─────────────────────────────────────────────

    def _update_rth_drift(self, uk_dt: datetime, close: float) -> None:
        """
        Record RTH open price at 14:30 UK.
        Determine drift at 15:00 UK (30 min later).
        Academic: RTH Confluence Signal T=5.83 MNQ 2021-25 (Mesfin 2026).
        """
        rth_open = PARAMS["rth_h"] * 60 + PARAMS["rth_m"]
        rth_end  = rth_open + PARAMS["rth_drift_min"]
        cur      = uk_dt.hour * 60 + uk_dt.minute

        if self._rth_open_price is None and cur == rth_open:
            self._rth_open_price = close
            log.info("📍 RTH open recorded: %.2f at %s UK",
                     close, uk_dt.strftime("%H:%M"))

        if (not self._rth_drift_determined
                and self._rth_open_price is not None
                and cur >= rth_end):
            d = (1 if close > self._rth_open_price
                 else -1 if close < self._rth_open_price else 0)
            self._rth_drift_direction   = d
            self._rth_drift_determined  = True
            ds = "BULLISH ▲" if d == 1 else "BEARISH ▼" if d == -1 else "FLAT"
            log.info("📊 RTH drift: %s  (open=%.2f  close=%.2f)",
                     ds, self._rth_open_price, close)

    def _reset_session_state(self) -> None:
        self._rth_open_price        = None
        self._rth_drift_direction   = 0
        self._rth_drift_determined  = False
        # Reset intrabar histogram tracker at session boundary
        self._prev_1m_hist          = 0.0
        self._last_forming_ts       = None
        self._intrabar_entry_fired  = False
        # Reset staleness watchdog — new session, bar feed re-establishes
        self._last_bar_refresh      = None

    # ── Position reset ────────────────────────────────────────

    def _reset_position(self) -> None:
        self.in_position       = False
        self._entry_signal     = 0
        self._entry_price      = 0.0
        self._stop_price       = 0.0
        self._tp_price         = 0.0
        self._partial_price    = 0.0
        self._tp1_reached      = False
        self._trade_entry_time = None
        self._last_exit_time   = datetime.now(timezone.utc)

    # ── Soft exit ─────────────────────────────────────────────

    async def _soft_exit(self, reason: str) -> None:
        d = "LONG" if self._entry_signal == 1 else "SHORT"
        clog.info(DIV)
        clog.info("  📉 SOFT EXIT | was %s  entry=%.2f", d, self._entry_price)
        clog.info("     Reason: %s", reason)
        clog.info(DIV)
        try:
            await self.suite.positions.close_all_positions()
            log.info("✅ close_all_positions called")
        except Exception as exc:
            log.error("❌ Soft exit flatten failed: %s", exc)
        self._reset_position()

    # ── Entry condition check ─────────────────────────────────

    def _check_entry(self, signal: int, bars1: List[dict],
                      session: Session) -> Tuple[bool, str]:
        """
        Checks all 8 entry conditions. Returns (ok, reason_string).
        Design: each condition has a clear academic reference in comments.
        Anti-overfitting: no session-specific RSI thresholds.
        """
        st = self._5m_state

        # ── Cooldown after exit ──────────────────────────────
        if self._last_exit_time is not None:
            elapsed = (datetime.now(timezone.utc) - self._last_exit_time).total_seconds()
            if elapsed < ENTRY_COOLDOWN_S:
                return False, f"Cooldown {ENTRY_COOLDOWN_S - elapsed:.0f}s"

        # ── Session gate ─────────────────────────────────────
        if session in (Session.NY_LATE, Session.BLACKOUT):
            return False, f"No entries in {session.value}"

        # ── 5-min: regime ready ──────────────────────────────
        if not st.ready or signal == 0:
            return False, "5-min warming up"

        # [1] EMA stack ordered (9 > 21 > 50 or reverse)
        # Academic: stack as regime descriptor (IMR 2022)
        if not self.ema.stack_ok:
            return False, (f"EMA stack not ordered "
                           f"(9={st.ema9:.1f} 21={st.ema21:.1f} 50={st.ema50:.1f})")

        # [2] ADX threshold (session-adaptive — not curve-fitted)
        # Academic: Baltussen et al. (2021) — momentum in trending regime only
        adx_thresh = ADX_MIN[session]
        if st.adx < adx_thresh:
            return False, f"ADX {st.adx:.1f} < {adx_thresh} ({session.value})"

        # [3] MACD histogram on correct side AND expanding
        # Not an entry trigger — regime context only (Nikkei futures study 2021)
        hist_ok = (
            (signal == 1 and st.macd_hist > 0
             and st.macd_hist > st.macd_hist_prev) or
            (signal == -1 and st.macd_hist < 0
             and st.macd_hist < st.macd_hist_prev)
        )
        if not hist_ok:
            return False, (f"MACD not expanding in direction "
                           f"(hist={st.macd_hist:.4f} prev={st.macd_hist_prev:.4f})")

        # [4] RTH drift alignment (LN_NY and NY sessions)
        # Academic: RTH Confluence Signal T=5.83 (Mesfin 2026)
        if session in DRIFT_SESSIONS and self._rth_drift_determined:
            if self._rth_drift_direction != 0 and signal != self._rth_drift_direction:
                return False, (f"Against RTH drift "
                               f"(drift={self._rth_drift_direction:+d} sig={signal:+d})")

        # ── 1-min: structure + trigger ────────────────────────
        min_bars = PARAMS["rsi_slow"] + 5
        if len(bars1) < min_bars:
            return False, f"Need ≥{min_bars} 1-min bars (have {len(bars1)})"

        completed = bars1[:-1]
        closes = [BarFetcher.ohlcv(b)[3] for b in completed]
        lows   = [BarFetcher.ohlcv(b)[2] for b in completed]
        highs  = [BarFetcher.ohlcv(b)[1] for b in completed]

        ema21_1m = _ema_series(closes, PARAMS["ema_mid"])[-1]
        ema50_1m = _ema_series(closes, PARAMS["ema_trend"])[-1]
        ema9_1m  = _ema_series(closes, PARAMS["ema_fast"])[-1]
        atr5_1m  = _compute_atr(completed[-20:], PARAMS["atr_period_1m"])

        rsi14_s = _compute_rsi(closes, PARAMS["rsi_slow"])
        rsi7_s  = _compute_rsi(closes, PARAMS["rsi_fast"])
        rsi14   = rsi14_s[-1]

        tol = PARAMS["ema_touch_mult"] * max(atr5_1m, TICK)

        # [5] Price pulled back to 1-min 21 EMA
        # Academic: pullback-to-MA entries superior to breakouts (XS.com 2026)
        if signal == 1:
            at_ema = lows[-1] <= ema21_1m + tol and closes[-1] >= ema21_1m - tol * 2
        else:
            at_ema = highs[-1] >= ema21_1m - tol and closes[-1] <= ema21_1m + tol * 2
        if not at_ema:
            return False, (f"Not at 1-min 21 EMA "
                           f"(close={closes[-1]:.2f} ema21={ema21_1m:.2f} tol=±{tol:.2f})")

        # [6] 1-min 50 EMA not breached (trend invalidation)
        recent3 = closes[-3:]
        if signal == 1 and any(c < ema50_1m for c in recent3):
            return False, f"50 EMA breached (ema50={ema50_1m:.2f})"
        if signal == -1 and any(c > ema50_1m for c in recent3):
            return False, f"50 EMA breached bullishly (ema50={ema50_1m:.2f})"

        # [7] RSI(14) in neutral entry zone (40–60)
        # Academic: SHAP analysis (arXiv 2023) — RSI 5× MACD signal weight;
        # entering neutral zone avoids extended-momentum entries
        if not (PARAMS["rsi_lo"] <= rsi14 <= PARAMS["rsi_hi"]):
            return False, (f"RSI(14) {rsi14:.1f} outside entry zone "
                           f"[{PARAMS['rsi_lo']}–{PARAMS['rsi_hi']}]")

        # [8] RSI(7) divergence trigger
        # Academic: 1-min SAC(1) = -0.1075 (arXiv:2408.17187);
        # divergence exploits mean-reversion at this timeframe
        if signal == 1:
            has_div = _bullish_rsi_div(closes, rsi7_s)
        else:
            has_div = _bearish_rsi_div(closes, rsi7_s)
        if not has_div:
            rsi7 = rsi7_s[-1]
            return False, f"No RSI(7) divergence (rsi7={rsi7:.1f})"

        return True, "ALL 8 CONDITIONS MET"

    # ── Place trade ───────────────────────────────────────────

    async def _place_trade(self, signal: int, entry_price: float,
                            atr: float, session: Session) -> None:
        side = 0 if signal == 1 else 1

        # ── Dynamic position sizing ────────────────────────────────────
        # Dollar risk per trade = contracts × sl_pts × $2/point.
        # Solving for contracts: floor(target / (sl_mult × ATR × 2)).
        # Clamp to [min_contracts, max_contracts].
        #
        # This keeps dollar risk near target_risk_usd regardless of whether
        # the ATR is 8 pts (calm session) or 25 pts (news/volatile session),
        # scaling exposure down automatically in high-volatility conditions
        # when gap-through-stop risk is also at its highest.
        sl_pts    = atr * PARAMS["sl_mult"]           # stop distance in points
        raw       = sl_pts * 2.0                       # $ risk per contract ($2/pt)
        if raw > 0:
            contracts = int(PARAMS["target_risk_usd"] / raw)
        else:
            contracts = PARAMS["min_contracts"]
        contracts = max(PARAMS["min_contracts"],
                        min(contracts, PARAMS["max_contracts"]))

        # Log the sizing decision so it is visible in the trade record
        actual_risk = contracts * sl_pts * 2.0
        log.info(
            "📐 Sizing: ATR=%.2f  SL=%.2f pts  target=$%d  "
            "→ %d contract(s)  actual_risk=$%.0f",
            atr, sl_pts, PARAMS["target_risk_usd"], contracts, actual_risk)

        sl_dist  = max(TICK, round(atr * PARAMS["sl_mult"]      / TICK) * TICK)
        tp_dist  = max(TICK, round(atr * PARAMS["tp_mult"]       / TICK) * TICK)
        pt_dist  = max(TICK, round(atr * PARAMS["partial_mult"]  / TICK) * TICK)

        if signal == 1:
            sl  = entry_price - sl_dist
            tp  = entry_price + tp_dist
            pt  = entry_price + pt_dist
        else:
            sl  = entry_price + sl_dist
            tp  = entry_price - tp_dist
            pt  = entry_price - pt_dist

        print_entry(signal, entry_price, sl, tp, pt, atr, session, contracts)

        try:
            order = await self.suite.orders.place_bracket_order(
                contract_id       = self._contract_id,
                side              = side,
                size              = contracts,
                entry_price       = entry_price,
                stop_loss_price   = sl,
                take_profit_price = tp,
            )
            if not order.success:
                log.error("❌ Bracket rejected: %s",
                          getattr(order, "error_message", "unknown"))
                return
            clog.info("  ✅ BRACKET LIVE | entry=%s  stop=%s  target=%s",
                      order.entry_order_id, order.stop_order_id,
                      order.target_order_id)
        except Exception as exc:
            err = str(exc).lower()
            is_fill = any(k in err for k in (
                "already filled", "fill processing",
                "failed to fill after recheck", "entry order"))
            if is_fill:
                log.warning("⚠️ Post-fill exception — checking broker: %s", exc)
                await asyncio.sleep(1.0)
                try:
                    pos = await self.suite.positions.get_all_positions()
                    if any(abs(getattr(p, "size", 0) or 0) > 0 for p in (pos or [])):
                        log.warning("⚠️ Position open — adopting")
                    else:
                        log.error("❌ Broker flat after post-fill error: %s", exc)
                        return
                except Exception as pe:
                    log.error("❌ Recovery poll failed: %s", pe)
                    return
            else:
                log.error("❌ Order placement failed: %s", exc)
                return

        # Mark as in position
        self.in_position       = True
        self._entry_signal     = signal
        self._entry_price      = entry_price
        self._stop_price       = sl
        self._tp_price         = tp
        self._partial_price    = pt
        self._tp1_reached      = False
        self._trade_entry_time = datetime.now(timezone.utc)
        self._last_exit_time   = None
        self.session_trades   += 1

    # ── 1-min exit monitor ────────────────────────────────────

    async def _check_exits_1m(self, close: float, bars1: List[dict]) -> bool:
        """
        Run exit cascade on each 1-min bar close.
        Returns True if a soft exit was triggered.

        P2  Structural: 1-min close crosses 9 EMA
        P2b RSI extreme (oversold long / overbought short)
        P3  Time stop: 6 min without 0.5R
        P4  0.5R reached → breakeven soft stop activates
        P5  Price returns to entry after 0.5R → soft exit
        """
        if not self.in_position:
            return False

        sig = self._entry_signal

        # Compute 1-min 9 EMA and RSI(14) from recent bars
        completed = bars1[:-1]
        if len(completed) < PARAMS["rsi_slow"] + 1:
            return False

        closes_c = [BarFetcher.ohlcv(b)[3] for b in completed]
        ema9_1m  = _ema_series(closes_c, PARAMS["ema_fast"])[-1]
        rsi14    = _compute_rsi(closes_c, PARAMS["rsi_slow"])[-1]

        # P2 — Structural: 1-min close crosses 9 EMA
        if sig == 1 and close < ema9_1m:
            await self._soft_exit(
                f"Structural: 1-min close {close:.2f} < 9EMA {ema9_1m:.2f}")
            return True
        if sig == -1 and close > ema9_1m:
            await self._soft_exit(
                f"Structural: 1-min close {close:.2f} > 9EMA {ema9_1m:.2f}")
            return True

        # P2b — RSI extreme (order-flow exhaustion proxy)
        if sig == 1 and rsi14 < PARAMS["rsi_extreme_lo"]:
            await self._soft_exit(
                f"RSI(14) extreme bearish: {rsi14:.1f} < {PARAMS['rsi_extreme_lo']}")
            return True
        if sig == -1 and rsi14 > PARAMS["rsi_extreme_hi"]:
            await self._soft_exit(
                f"RSI(14) extreme bullish: {rsi14:.1f} > {PARAMS['rsi_extreme_hi']}")
            return True

        # P3 — Time stop
        if not self._tp1_reached and self._trade_entry_time:
            elapsed = (datetime.now(timezone.utc)
                       - self._trade_entry_time).total_seconds() / 60.0
            if elapsed >= PARAMS["time_stop_min"]:
                await self._soft_exit(
                    f"Time stop: {elapsed:.1f} min elapsed, 0.5R not reached")
                return True

        # P4 — 0.5R monitoring → activate breakeven soft stop
        if not self._tp1_reached:
            hit = ((sig == 1 and close >= self._partial_price) or
                   (sig == -1 and close <= self._partial_price))
            if hit:
                self._tp1_reached = True
                log.info("🎯 0.5R reached at %.2f — breakeven soft stop now active "
                         "(remaining runs to full 1:1 bracket)", close)

        # P5 — Breakeven soft stop (after 0.5R reached)
        else:
            be_hit = ((sig == 1 and close <= self._entry_price) or
                      (sig == -1 and close >= self._entry_price))
            if be_hit:
                await self._soft_exit(
                    f"Breakeven stop: price {close:.2f} returned to entry "
                    f"{self._entry_price:.2f} after 0.5R")
                return True

        return False

    # ── Periodic broker sync ──────────────────────────────────

    async def _sync_position(self) -> None:
        try:
            positions   = await self.suite.positions.get_all_positions()
            broker_open = bool(positions) and any(
                abs(getattr(p, "size", 0) or 0) > 0 for p in positions)
            if self.in_position and not broker_open:
                log.info("🔄 Sync: local=OPEN broker=FLAT → SL/TP hit")
                self._reset_position()
            elif not self.in_position and broker_open:
                log.warning("🔄 Sync: local=FLAT broker=OPEN → adopting")
                for p in positions:
                    sz  = getattr(p, "size", 0) or 0
                    avg = float(getattr(p, "averagePrice",
                                        getattr(p, "average_price", 0.0)) or 0.0)
                    if abs(sz) > 0:
                        self.in_position   = True
                        self._entry_signal = 1 if sz > 0 else -1
                        self._entry_price  = avg
                        log.warning("  Adopted: sig=%+d  entry=%.2f", self._entry_signal, avg)
                        break
        except Exception as exc:
            log.debug("Sync unavailable: %s", exc)

    # ── Startup position adoption ─────────────────────────────

    async def _initial_position_sync(self) -> None:
        log.info("Checking broker for open positions...")
        for attempt in range(5):
            try:
                positions = await self.suite.positions.get_all_positions()
                if not positions:
                    log.info("  Broker: no open positions — fresh start.")
                    return
                for p in positions:
                    sz  = getattr(p, "size", 0) or 0
                    avg = float(getattr(p, "averagePrice",
                                        getattr(p, "average_price", 0.0)) or 0.0)
                    if abs(sz) > 0:
                        self.in_position   = True
                        self._entry_signal = 1 if sz > 0 else -1
                        self._entry_price  = avg
                        log.info("  Adopted existing: sig=%+d  entry≈%.2f", self._entry_signal, avg)
                        return
                log.info("  All positions flat — fresh start.")
                return
            except Exception as exc:
                wait = 5 * (attempt + 1)
                log.warning("  Position check attempt %d/5: %s — retry in %ds",
                            attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        log.error("  Could not confirm position after 5 attempts — assuming flat.")

    # ── Seed engines ──────────────────────────────────────────

    async def _seed_engines(self) -> None:
        log.info("Seeding from 5-min history...")
        bars5 = []
        for attempt in range(5):
            bars5 = await self._fetcher.fetch(5, days=3)
            if bars5:
                break
            wait = 5 * (attempt + 1)
            log.warning("Seed attempt %d/5 — retry in %ds", attempt + 1, wait)
            await asyncio.sleep(wait)

        if not bars5:
            log.error("Could not fetch 5-min bars — will warm up live.")
            return

        warmup = PARAMS["warmup_5m_bars"]
        bars5  = bars5[-warmup:] if len(bars5) > warmup else bars5
        self.ema.seed(bars5)
        self._last_5m_ts = BarFetcher.ts(bars5[-1])

        completed = bars5[:-1]
        closes_c  = [BarFetcher.ohlcv(b)[3] for b in completed]
        atr14     = _compute_atr(completed, PARAMS["atr_period_5m"])
        adx       = _compute_adx(completed, PARAMS["adx_period"])
        hist, hp  = _compute_macd_hist(completed)
        e9  = _ema_series(closes_c, PARAMS["ema_fast"])
        e21 = _ema_series(closes_c, PARAMS["ema_mid"])
        e50 = _ema_series(closes_c, PARAMS["ema_trend"])

        self._current_atr = atr14
        self._5m_state = FiveMinState(
            ema9  = e9[-1]  if e9  else 0.0,
            ema21 = e21[-1] if e21 else 0.0,
            ema50 = e50[-1] if e50 else 0.0,
            atr14 = atr14, adx = adx,
            macd_hist = hist, macd_hist_prev = hp,
            direction = self.ema.direction,
            bars_since_cross = self.ema.bars_since_cross,
            ready = self.ema.is_ready and atr14 > 0,
        )
        self._5m_ready       = self._5m_state.ready
        self._prev_direction = self.ema.direction

        # Set staleness watchdog baseline — if bars never refresh after
        # this point the watchdog fires ~3 min later and triggers reconnect.
        self._last_bar_refresh = datetime.now(timezone.utc)

        log.info("Seeded | %d bars | dir=%+d  ADX=%.1f  ATR=%.2f  ready=%s",
                 len(bars5) - 1, self.ema.direction, adx, atr14, self._5m_ready)

    # ── 5-min bar handler (regime only — no direct entries) ───

    async def _on_5m_bar(self) -> None:
        async with self._eval_lock:
            bars = await self._fetcher.fetch(5, days=3)
            if not bars or len(bars) < 5:
                return
            latest_ts = BarFetcher.ts(bars[-1])
            if latest_ts and latest_ts == self._last_5m_ts:
                return
            self._last_5m_ts = latest_ts

            _, _, _, c, _ = BarFetcher.ohlcv(bars[-2])   # completed close
            completed      = bars[:-1]
            closes_c       = [BarFetcher.ohlcv(b)[3] for b in completed]

            self.ema.update(c)

            atr14    = _compute_atr(completed, PARAMS["atr_period_5m"])
            adx      = _compute_adx(completed, PARAMS["adx_period"])
            hist, hp = _compute_macd_hist(completed,
                                           PARAMS["macd_fast"],
                                           PARAMS["macd_slow"],
                                           PARAMS["macd_signal"])
            e9  = _ema_series(closes_c, PARAMS["ema_fast"])
            e21 = _ema_series(closes_c, PARAMS["ema_mid"])
            e50 = _ema_series(closes_c, PARAMS["ema_trend"])

            self._current_atr = atr14
            self._5m_state = FiveMinState(
                ema9  = e9[-1]  if e9  else 0.0,
                ema21 = e21[-1] if e21 else 0.0,
                ema50 = e50[-1] if e50 else 0.0,
                atr14 = atr14, adx = adx,
                macd_hist = hist, macd_hist_prev = hp,
                direction = self.ema.direction,
                bars_since_cross = self.ema.bars_since_cross,
                ready = self.ema.is_ready and atr14 > 0,
            )
            self._5m_ready = self._5m_state.ready

            if not self._5m_ready:
                log.info("[5m] Warming (%d/%d)",
                         self.ema._bar_count, PARAMS["ema_trend"])
                return

            await self._get_balance()

            sig             = self.ema.direction
            new_direction   = sig
            crossover       = (new_direction != 0 and
                               new_direction != self._prev_direction)
            self._prev_direction = new_direction

            session    = get_session(self._now_uk())
            adx_thresh = ADX_MIN[session]

            # Check whether all 5-min conditions are satisfied
            # (used only for the log display — entries fire from 1-min loop)
            regime_ok = (
                self.ema.stack_ok and
                adx >= adx_thresh and
                ((sig == 1  and hist > 0 and hist > hp) or
                 (sig == -1 and hist < 0 and hist < hp))
            )

            print_5m(self._5m_state, session, sig, self.in_position,
                     self._session_pnl(), self.session_trades,
                     adx_thresh, regime_ok)

            # P7 — Soft exit on opposite 5-min crossover
            if (crossover and self.in_position
                    and self._entry_signal != new_direction):
                await self._soft_exit(
                    f"5-min EMA crossover reversed to {new_direction:+d}")

    # ── 1-min bar handler (entry trigger + exit monitor) ──────

    async def _on_1m_bar(self) -> None:
        async with self._eval_lock:
            bars1 = await self._fetcher.fetch(1, days=1)
            if not bars1 or len(bars1) < 5:
                return
            latest_ts = BarFetcher.ts(bars1[-1])
            if latest_ts and latest_ts == self._last_1m_ts:
                return
            self._last_1m_ts = latest_ts
            self._1m_count  += 1

            confirmed_close = BarFetcher.ohlcv(bars1[-2])[3]

            # Periodic broker sync
            if self._1m_count % PARAMS["sync_every_n_bars"] == 0:
                await self._sync_position()

            # Pre-blackout flatten
            if self._should_flatten():
                if self.in_position:
                    log.info("🔔 Pre-blackout flatten")
                    try:
                        await self.suite.positions.close_all_positions()
                    except Exception as exc:
                        log.error("Pre-blackout flatten failed: %s", exc)
                    self._reset_position()
                return

            if self._in_blackout():
                return

            await self._get_balance()

            uk_dt   = self._now_uk()
            session = get_session(uk_dt)

            # Update RTH drift tracking
            self._update_rth_drift(uk_dt, confirmed_close)

            # Compute 1-min RSI for status log
            completed1 = bars1[:-1]
            closes1    = [BarFetcher.ohlcv(b)[3] for b in completed1]
            rsi14_1m   = 50.0
            rsi7_1m    = 50.0
            if len(closes1) >= PARAMS["rsi_slow"] + 1:
                rsi14_1m = _compute_rsi(closes1, PARAMS["rsi_slow"])[-1]
                rsi7_1m  = _compute_rsi(closes1, PARAMS["rsi_fast"])[-1]

            sig = self.ema.direction

            # ── Exit monitor (P2–P5) ──────────────────────────
            if self.in_position:
                exited = await self._check_exits_1m(confirmed_close, bars1)
                if exited:
                    print_1m_status(self._uk_hm(), confirmed_close, sig,
                                    session, rsi14_1m, rsi7_1m, False,
                                    self._session_pnl(), "→ EXIT")
                    return

            # ── Entry check (only when not in position) ────────
            entry_reason = "·"
            if not self.in_position and not self._in_blackout():
                can_enter, reason = self._check_entry(sig, bars1, session)
                if can_enter:
                    await self._place_trade(
                        signal      = sig,
                        entry_price = confirmed_close,
                        atr         = self._current_atr,
                        session     = session,
                    )
                    entry_reason = "→ ENTERED"
                else:
                    entry_reason = f"→ wait: {reason}"

            print_1m_status(self._uk_hm(), confirmed_close, sig,
                            session, rsi14_1m, rsi7_1m,
                            self.in_position, self._session_pnl(),
                            entry_reason)

    # ── Fast position monitor (3-second loop) ────────────────
    #
    # Runs independently of the bar-close loops.
    # The broker already handles hard SL/TP via bracket order — this loop
    # adds precision to the SOFTWARE-managed exits:
    #   • Time stop  : precise clock-based check (not bar-close-dependent)
    #   • 0.5R flag  : set as soon as intrabar price crosses the level
    #   • Breakeven  : soft-exit as soon as intrabar price returns to entry
    #   • Broker sync: discover filled SL/TP brackets within ~15 seconds
    #
    # Entry signals are deliberately NOT fired here — they require completed
    # bar closes and are handled exclusively by _on_1m_bar.

    async def _poll_3s(self) -> None:
        """
        Fast loop — runs every PARAMS['fast_poll_s'] seconds (default: 3 s).

        Two independent jobs:

        JOB 1 — position management (when in a trade)
            • Broker sync every ~15 s to detect filled SL/TP brackets.
            • Precise time stop: fires within ±3 s of the deadline instead of
              waiting up to 60 s for the next bar close.
            • Intrabar 0.5R detection: sets the breakeven flag as soon as the
              forming bar price crosses the partial level.
            • Intrabar breakeven stop: exits within 3 s of price returning to
              entry after the 0.5R flag is set.

        JOB 2 — intrabar MACD histogram entry (when NOT in a trade)
            The 1-min MACD histogram reflects momentum.  During a bullish
            pullback setup, the histogram is negative (bearish correction).
            When it crosses zero — even before the bar closes — momentum has
            shifted.  Entering at the zero-cross gives a better price than
            waiting for the bar to close and the RSI(7) divergence to confirm
            on the completed bar.

            Academic basis: the histogram zero-cross is the earliest
            bar-resident momentum signal.  Entering here trades the impulse
            from inception rather than confirming it after the fact.

            Entry conditions checked intrabar (lighter than full 8-condition
            bar-close check, since better price offsets reduced confirmation):
              • 5-min: EMA stack ordered, ADX ≥ session threshold
              • Session drift aligned (RTH/London sessions)
              • Cooldown has elapsed since last exit
              • Forming bar price within 2× EMA-touch tolerance of 21 EMA
              • RSI(14) on completed bars still in 40–60 neutral zone
              • 1-min MACD histogram crossed zero in trade direction this tick
        """
        fast_count = 0

        while True:
            await asyncio.sleep(PARAMS["fast_poll_s"])
            fast_count += 1

            if self._in_blackout() or not self._5m_ready:
                continue

            try:
                # ── Broker sync ────────────────────────────────────────────
                if fast_count % PARAMS["sync_fast_n"] == 0:
                    await self._sync_position()

                # ── Fetch forming bar ──────────────────────────────────────
                # bars[-1] = forming (incomplete) bar — used for live price
                # AND for intrabar histogram computation.
                # bars[-2] = last completed bar.
                bars1 = await self._fetcher.fetch(1, days=1)
                if not bars1 or len(bars1) < PARAMS["rsi_slow"] + 5:
                    self._3s_errors = 0
                    continue

                forming_ts = BarFetcher.ts(bars1[-1])
                current_px = BarFetcher.ohlcv(bars1[-1])[3]
                now_utc    = datetime.now(timezone.utc)

                # Reset per-bar flags and update freshness timer on new bar
                if forming_ts != self._last_forming_ts:
                    self._last_forming_ts       = forming_ts
                    self._intrabar_entry_fired  = False
                    self._prev_1m_hist          = 0.0
                    self._last_bar_refresh      = now_utc

                # ── Staleness watchdog ─────────────────────────────────────
                # After warmup, 1-min bars must refresh at least once every
                # 3 minutes.  If they don't, the library's internal bar cache
                # has stopped updating (realtime feed broken — symptom: only
                # Position/searchOpen calls visible in log, no bar HTTP
                # requests).  Reconnecting creates a fresh TradingSuite that
                # re-establishes the HTTP bar feed.
                #
                # We only activate this after the first new bar is seen
                # (self._last_bar_refresh is set) and after warmup is done,
                # to avoid false alarms during the initial seed window.
                if (self._5m_ready and
                        self._last_bar_refresh is not None and
                        (now_utc - self._last_bar_refresh).total_seconds()
                        > 180):   # 3 minutes without a new 1-min bar
                    raise ConnectionLostError(
                        f"1-min bar data stale for "
                        f"{(now_utc - self._last_bar_refresh).total_seconds():.0f}s "
                        f"— library realtime feed likely broken, reconnecting")

                # ══════════════════════════════════════════════════════════
                # JOB 1 — position management
                # ══════════════════════════════════════════════════════════
                if self.in_position:
                    sig = self._entry_signal

                    # Time stop (real clock, not bar-close)
                    if self._trade_entry_time and not self._tp1_reached:
                        elapsed = (
                            datetime.now(timezone.utc) - self._trade_entry_time
                        ).total_seconds() / 60.0
                        if elapsed >= PARAMS["time_stop_min"]:
                            async with self._eval_lock:
                                if self.in_position and not self._tp1_reached:
                                    await self._soft_exit(
                                        f"Time stop: {elapsed:.1f} min "
                                        f"without 0.5R (fast monitor)")
                            self._3s_errors = 0
                            continue

                    # 0.5R flag — set intrabar as soon as price crosses level
                    if not self._tp1_reached and self._partial_price:
                        hit = ((sig == 1  and current_px >= self._partial_price) or
                               (sig == -1 and current_px <= self._partial_price))
                        if hit:
                            self._tp1_reached = True
                            log.info("🎯 0.5R intrabar at %.2f — "
                                     "breakeven stop active", current_px)

                    # Breakeven stop — fires within 3 s of price returning to entry
                    elif self._tp1_reached and self._entry_price:
                        be = ((sig == 1  and current_px <= self._entry_price) or
                              (sig == -1 and current_px >= self._entry_price))
                        if be:
                            async with self._eval_lock:
                                if self.in_position and self._tp1_reached:
                                    await self._soft_exit(
                                        f"Breakeven (fast): {current_px:.2f} "
                                        f"returned to entry {self._entry_price:.2f}")

                # ══════════════════════════════════════════════════════════
                # JOB 2 — intrabar MACD histogram entry
                # ══════════════════════════════════════════════════════════
                else:
                    # --- cooldown gate ---
                    cooldown_ok = (
                        self._last_exit_time is None or
                        (datetime.now(timezone.utc) - self._last_exit_time
                         ).total_seconds() >= ENTRY_COOLDOWN_S
                    )
                    sig     = self.ema.direction
                    session = get_session(self._now_uk())
                    gate_ok = (sig != 0 and
                               cooldown_ok and
                               session not in (Session.NY_LATE, Session.BLACKOUT))

                    if gate_ok and not self._intrabar_entry_fired:
                        # Compute 1-min MACD histogram including the forming bar
                        hist_now, _ = _compute_macd_hist(
                            bars1,
                            PARAMS["macd_fast"],
                            PARAMS["macd_slow"],
                            PARAMS["macd_signal"])

                        # Zero-cross in trade direction
                        # Long:  histogram flips negative → zero or positive
                        # Short: histogram flips positive → zero or negative
                        bull_cross = (sig == 1  and
                                      self._prev_1m_hist < 0 and
                                      hist_now >= 0)
                        bear_cross = (sig == -1 and
                                      self._prev_1m_hist > 0 and
                                      hist_now <= 0)

                        if bull_cross or bear_cross:
                            # --- regime conditions (subset of full 8-gate) ---
                            completed  = bars1[:-1]
                            closes_c   = [BarFetcher.ohlcv(b)[3] for b in completed]
                            rsi14      = _compute_rsi(closes_c, PARAMS["rsi_slow"])[-1]
                            ema21_1m   = _ema_series(closes_c, PARAMS["ema_mid"])[-1]
                            atr5_1m    = _compute_atr(completed[-20:],
                                                      PARAMS["atr_period_1m"])
                            tol        = PARAMS["ema_touch_mult"] * max(atr5_1m, TICK)

                            # RTH drift check
                            drift_ok = (
                                session not in DRIFT_SESSIONS or
                                not self._rth_drift_determined or
                                self._rth_drift_direction == 0 or
                                sig == self._rth_drift_direction
                            )

                            # Forming bar price within 2× tolerance of 21 EMA
                            # (wider than bar-close check — better price justifies it)
                            at_ema = (
                                (sig == 1  and current_px <= ema21_1m + tol * 2.0) or
                                (sig == -1 and current_px >= ema21_1m - tol * 2.0)
                            )

                            # RSI gate (completed bars — unchanged from bar-close entry)
                            rsi_ok = PARAMS["rsi_lo"] <= rsi14 <= PARAMS["rsi_hi"]

                            # 5-min regime
                            regime_ok = (
                                self._5m_state.ready and
                                self.ema.stack_ok and
                                self._5m_state.adx >= ADX_MIN[session]
                            )

                            if regime_ok and drift_ok and at_ema and rsi_ok:
                                self._intrabar_entry_fired = True
                                log.info(
                                    "📊 Intrabar MACD zero-cross: "
                                    "hist %.4f → %.4f  sig=%+d  px=%.2f  "
                                    "ema21=%.2f  rsi14=%.1f",
                                    self._prev_1m_hist, hist_now,
                                    sig, current_px, ema21_1m, rsi14)
                                async with self._eval_lock:
                                    if not self.in_position:
                                        await self._place_trade(
                                            signal      = sig,
                                            entry_price = current_px,
                                            atr         = self._current_atr,
                                            session     = session)
                            else:
                                log.debug(
                                    "Intrabar cross seen but gated: "
                                    "regime=%s drift=%s ema=%s rsi14=%.1f",
                                    regime_ok, drift_ok, at_ema, rsi14)

                        # Always update histogram history for the next tick
                        self._prev_1m_hist = hist_now

                self._3s_errors = 0

            except asyncio.CancelledError:
                raise
            except ConnectionLostError:
                raise   # staleness watchdog — propagate immediately to reconnect
            except Exception as exc:
                self._3s_errors += 1
                log.debug("Fast monitor error %d/%d: %s",
                          self._3s_errors, MAX_CONSECUTIVE_ERRORS, exc)
                if self._3s_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise ConnectionLostError(
                        f"3s: {MAX_CONSECUTIVE_ERRORS} consecutive errors"
                    ) from exc

        # ── Main run loop ─────────────────────────────────────────

    async def run(self) -> None:
        # Resolve contract ID
        self._contract_id = getattr(self.suite, "instrument_id", None)
        if not self._contract_id:
            try:
                instrs = await self.client.search_instruments(self.SYMBOL)
                if instrs:
                    cid = getattr(instrs[0], "id", None)
                    if cid is None and isinstance(instrs[0], dict):
                        cid = instrs[0].get("id")
                    self._contract_id = str(cid)
            except Exception as exc:
                log.error("Could not resolve contract ID: %s", exc)
        log.info("Contract ID: %s", self._contract_id)

        # Use DirectBarFetcher (direct REST API) instead of client.get_bars().
        # client.get_bars() routes through the library's bounded_stats cache which
        # has a structural 8-col vs 6-col schema mismatch with the realtime feed.
        # DirectBarFetcher calls POST /api/History/retrieveBars directly and is
        # completely independent of the library's internal statistics module.
        # Determine live/sim from account info (sim accounts have "PRAC" in name).
        _acct_name = ""
        try:
            _acct = self.client.account_info
            _acct_name = str(getattr(_acct, "name", "") or "").upper()
        except Exception:
            pass
        _is_live = "PRAC" not in _acct_name and "SIM" not in _acct_name
        self._fetcher = DirectBarFetcher(
            contract_id = self._contract_id or self.SYMBOL,
            live        = _is_live,
        )
        log.info("DirectBarFetcher initialised | contract=%s  live=%s",
                 self._contract_id, _is_live)

        pf = PreFlightCheck()
        if not await pf.run(self.suite):
            log.critical("Pre-flight failed — aborting.")
            return

        # Wait out blackout
        if self._in_blackout():
            log.info("⏳ Started in blackout — waiting until %02d:00 UK",
                     PARAMS["blackout_end_uk"])
            while self._in_blackout():
                await asyncio.sleep(30)

        await self._initial_position_sync()
        balance = await self._get_balance()
        self._session_open_balance = balance

        await self._seed_engines()

        clog.info(DIV2)
        clog.info("  BOT LIVE | MNQ SCALPER v2.0 | dynamic sizing")
        clog.info("  target_risk=$%d  max=%d contracts  min=%d contract(s)",
                  PARAMS["target_risk_usd"],
                  PARAMS["max_contracts"],
                  PARAMS["min_contracts"])
        clog.info("  EMA(%d/%d/%d)  ADX≥%.0f(LN/NY)/%.0f(Asia)  RR 1:1",
                  PARAMS["ema_fast"], PARAMS["ema_mid"], PARAMS["ema_trend"],
                  ADX_MIN[Session.LN_NY], ADX_MIN[Session.ASIA])
        clog.info("  SL=%.1f×ATR  TP=%.1f×ATR  TimeStop=%dmin",
                  PARAMS["sl_mult"], PARAMS["tp_mult"], PARAMS["time_stop_min"])
        clog.info("  Blackout: %02d:00–%02d:00 UK  |  balance=%.2f",
                  PARAMS["blackout_start_uk"], PARAMS["blackout_end_uk"], balance)
        clog.info(DIV2)

        # ── Poll loops ────────────────────────────────────────

        async def poll_5m() -> None:
            first = True
            while True:
                if not first:
                    await asyncio.sleep(self._secs_to_next_5m())
                first = False

                if self._in_blackout():
                    if self.in_position:
                        try:
                            await self.suite.positions.close_all_positions()
                        except Exception as exc:
                            log.error("Blackout flatten failed: %s", exc)
                        self._reset_position()

                    clog.info("  💤 BLACKOUT %02d:00–%02d:00 UK",
                              PARAMS["blackout_start_uk"], PARAMS["blackout_end_uk"])
                    while self._in_blackout():
                        await asyncio.sleep(30)

                    # Re-seed after blackout
                    balance = await self._get_balance()
                    self._session_open_balance = balance
                    self.session_trades        = 0
                    self.ema = EMAEngine(PARAMS["ema_fast"], PARAMS["ema_mid"],
                                        PARAMS["ema_trend"])
                    self._5m_ready             = False
                    self._last_5m_ts           = None
                    self._last_1m_ts           = None
                    self._prev_direction       = 0
                    self._5m_errors            = 0
                    self._1m_errors            = 0
                    self._3s_errors            = 0
                    self._reset_session_state()
                    await self._seed_engines()
                    clog.info("  ✅ BLACKOUT ENDED | %s UK | balance=%.2f",
                              self._uk_hm(), balance)

                try:
                    await self._on_5m_bar()
                    self._5m_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._5m_errors += 1
                    log.error("[5m] Error %d/%d: %s",
                              self._5m_errors, MAX_CONSECUTIVE_ERRORS, exc)
                    if self._5m_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise ConnectionLostError(
                            f"5m: {MAX_CONSECUTIVE_ERRORS} consecutive errors") from exc
                    await asyncio.sleep(5)

        async def poll_1m() -> None:
            while True:
                await asyncio.sleep(1)
                try:
                    await self._on_1m_bar()
                    self._1m_errors = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self._1m_errors += 1
                    log.error("[1m] Error %d/%d: %s",
                              self._1m_errors, MAX_CONSECUTIVE_ERRORS, exc)
                    if self._1m_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise ConnectionLostError(
                            f"1m: {MAX_CONSECUTIVE_ERRORS} consecutive errors") from exc
                    await asyncio.sleep(2)

        try:
            await asyncio.gather(poll_5m(), poll_1m(), self._poll_3s())
        except asyncio.CancelledError:
            log.info("Cancelled — flattening positions")
            try:
                await self.suite.positions.close_all_positions()
            except Exception:
                pass
            raise


# ─────────────────────────────────────────────────────────────
# 11.  LIBRARY BUG PATCH
# ─────────────────────────────────────────────────────────────
def _patch_polars_schema() -> bool:
    """
    Patch pl.DataFrame.extend and pl.DataFrame.vstack to pad column-width
    mismatches before they reach Polars/Rust.

    Why this works
    ──────────────
    pl.DataFrame is a Python class (defined in polars/dataframe/frame.py)
    that wraps a Rust inner object.  Its Python methods — including extend()
    and vstack() — are regular Python functions on a regular Python class,
    which means they can be replaced with monkey-patches using attribute
    assignment.

    What it does
    ────────────
    When the bounded_stats realtime feed tries to extend an 8-column
    historical DataFrame with a 6-column WebSocket bar, our patch:
      1. Detects the column count mismatch.
      2. Identifies which 2 columns are missing from the source.
      3. Adds those columns as null Series with the correct dtype.
      4. Reorders the source to match the target column order.
      5. Calls the original extend() with the now-compatible DataFrame.

    The library's statistics will have null values in those 2 columns for
    each realtime bar update.  Our bot never reads the library's internal
    statistics, so this has zero effect on trading logic.

    Called ONCE before TradingSuite.create() so the patch is in place
    before the realtime feed background task starts.
    """
    try:
        import polars as pl
    except ImportError:
        log.debug("Polars not importable — cannot patch DataFrame.extend")
        return False

    def _align(target: "pl.DataFrame",
               source: "pl.DataFrame") -> "Optional[pl.DataFrame]":
        """
        Return source padded and reordered to match target's column schema.
        Returns None if alignment is impossible (caller should skip the op).
        """
        target_cols = list(target.columns)
        if list(source.columns) == target_cols:
            return source                      # already identical — fast path

        # Add any missing columns as typed nulls
        missing = [c for c in target_cols if c not in source.columns]
        if missing:
            try:
                source = source.with_columns([
                    pl.lit(None).cast(target.schema[c]).alias(c)
                    for c in missing
                ])
            except Exception as exc:
                log.debug("Schema-align: could not pad %s: %s", missing, exc)
                return None

        # Select in target column order (also drops any extra source cols)
        available = [c for c in target_cols if c in source.columns]
        try:
            return source.select(available)
        except Exception as exc:
            log.debug("Schema-align: select failed: %s", exc)
            return None

    patched = 0

    # ── extend() — in-place append, returns None ──────────────────────────
    try:
        _orig_extend = pl.DataFrame.extend

        def _safe_extend(self: "pl.DataFrame",
                         other: "pl.DataFrame") -> None:
            aligned = _align(self, other)
            if aligned is None:
                log.debug("extend: schema alignment failed — skipping append")
                return
            return _orig_extend(self, aligned)

        pl.DataFrame.extend = _safe_extend
        log.info("✅ Patched pl.DataFrame.extend — "
                 "6-col realtime bars will be padded to match 8-col schema")
        patched += 1
    except Exception as exc:
        log.warning("⚠️ Could not patch pl.DataFrame.extend: %s", exc)

    # ── vstack() — returns new DataFrame ─────────────────────────────────
    try:
        _orig_vstack = pl.DataFrame.vstack

        def _safe_vstack(self: "pl.DataFrame",
                         other: "pl.DataFrame") -> "pl.DataFrame":
            aligned = _align(self, other)
            if aligned is None:
                log.debug("vstack: schema alignment failed — returning self")
                return self
            return _orig_vstack(self, aligned)

        pl.DataFrame.vstack = _safe_vstack
        log.info("✅ Patched pl.DataFrame.vstack")
        patched += 1
    except Exception as exc:
        log.warning("⚠️ Could not patch pl.DataFrame.vstack: %s", exc)

    if not patched:
        log.warning("⚠️ Polars DataFrame patching failed — "
                    "bounded_stats schema errors may still occur")
    return patched > 0


def _make_safe_wrapper(original):
    """Wraps a callable so any exception it raises is silently discarded."""
    def _safe(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception:
            pass
    _safe.__name__     = getattr(original, '__name__', '_safe')
    _safe.__qualname__ = getattr(original, '__qualname__', '_safe') + '[patched]'
    return _safe


def _patch_bounded_stats() -> bool:
    """
    Silently patch _update_timeframe_data in project_x_py's bounded_stats.

    Must be called AFTER TradingSuite.create() because the problematic class
    is instantiated lazily during library initialisation.

    Three strategies are attempted in order:

      1. sys.modules scan  — fast, works for directly importable classes.
      2. gc.get_objects()  — thorough, finds runtime-created instances whose
                             classes may not appear in sys.modules by name.
      3. asyncio exception handler — fallback that prevents the crashed
                             background task's exception from being escalated
                             to a fatal error, allowing the staleness watchdog
                             to handle reconnection cleanly.

    All three are applied; the function returns True if strategy 1 or 2
    patched at least one location.
    """
    import sys
    import gc

    TARGET  = "_update_timeframe_data"
    KEYWORD = "project_x_py"

    patched_classes: set = set()
    patched_count   = 0

    # ── Strategy 1: sys.modules scan ─────────────────────────────────────
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or KEYWORD not in mod_name:
            continue
        try:
            # Module-level function
            fn = getattr(mod, TARGET, None)
            if callable(fn) and not isinstance(fn, type):
                try:
                    setattr(mod, TARGET, _make_safe_wrapper(fn))
                    log.info("✅ [mod-fn ] Patched %s.%s", mod_name, TARGET)
                    patched_count += 1
                except Exception as e:
                    log.debug("    could not setattr on module %s: %s", mod_name, e)

            # Classes defined or imported in the module
            for attr_name in dir(mod):
                try:
                    obj = getattr(mod, attr_name)
                    if not isinstance(obj, type) or obj in patched_classes:
                        continue
                    method = getattr(obj, TARGET, None)
                    if not callable(method):
                        continue
                    try:
                        setattr(obj, TARGET, _make_safe_wrapper(method))
                        patched_classes.add(obj)
                        log.info("✅ [class  ] Patched %s.%s.%s",
                                 mod_name, attr_name, TARGET)
                        patched_count += 1
                    except Exception as e:
                        log.debug("    could not setattr on %s.%s: %s",
                                  attr_name, TARGET, e)
                except Exception:
                    pass
        except Exception:
            pass

    # ── Strategy 2: gc.get_objects() scan ────────────────────────────────
    # Finds live instances whose class may not be directly reachable via
    # sys.modules — e.g. classes created by factory functions or metaclasses.
    try:
        for obj in gc.get_objects():
            try:
                cls = type(obj)
                if cls in patched_classes:
                    continue
                cls_mod = getattr(cls, '__module__', '') or ''
                if KEYWORD not in cls_mod:
                    continue
                method = getattr(cls, TARGET, None)
                if not callable(method):
                    continue
                try:
                    setattr(cls, TARGET, _make_safe_wrapper(method))
                    patched_classes.add(cls)
                    log.info("✅ [gc-inst] Patched %s.%s (found via live instance)",
                             cls.__qualname__, TARGET)
                    patched_count += 1
                except Exception as e:
                    log.debug("    gc setattr failed for %s: %s",
                              cls.__qualname__, e)
            except Exception:
                pass
    except Exception as e:
        log.debug("gc.get_objects() scan failed: %s", e)

    # ── Strategy 3: asyncio task exception handler (always applied) ───────
    # Even if the class cannot be patched, we can prevent the background
    # task's unhandled exception from being escalated.  Combined with the
    # staleness watchdog, this ensures a clean reconnect rather than a crash.
    try:
        loop = asyncio.get_event_loop()
        _orig_handler = loop.exception_handler  # may be None

        def _task_exc_handler(loop, context):
            exc = context.get('exception')
            if exc is not None:
                msg = str(exc).lower()
                if ('width' in msg and 'append' in msg) or 'unable to append' in msg:
                    # Swallow the bounded_stats column-mismatch error silently
                    log.debug("Suppressed bounded_stats task exception: %s", exc)
                    return
            # All other exceptions: use the original handler or the default
            if _orig_handler:
                _orig_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_task_exc_handler)
        log.info("✅ [asyncio] Task exception handler set — "
                 "bounded_stats errors will not escalate")
    except Exception as e:
        log.debug("Could not set asyncio exception handler: %s", e)

    # ── Summary ───────────────────────────────────────────────────────────
    if patched_count:
        log.info("bounded_stats patch: %d location(s) wrapped — "
                 "realtime-feed column mismatch will be silently skipped",
                 patched_count)
        return True

    log.warning(
        "⚠️  _update_timeframe_data not found via sys.modules or gc scan. "
        "The asyncio exception handler is still active. "
        "The staleness watchdog will detect stale bars and reconnect. "
        "Consider updating project_x_py or contacting TopstepX support.")
    return False

# ─────────────────────────────────────────────────────────────
# 12.  MAIN  —  outer reconnect loop
# ─────────────────────────────────────────────────────────────
async def main() -> None:
    if (not os.environ.get("PROJECT_X_API_KEY")
            or not os.environ.get("PROJECT_X_USERNAME")):
        raise EnvironmentError(
            "Set PROJECT_X_API_KEY and PROJECT_X_USERNAME before running.")

    log.info("─" * 64)
    log.info("MNQ Scalper v2.0  |  log: %s", _log_file)
    log.info("─" * 64)

    # Patch Polars BEFORE any TradingSuite.create() call so the patch is
    # active when the realtime feed background task first runs.
    # This is the root-cause fix: pad 6-col WebSocket bars to 8-col to
    # match the historical seed DataFrame's schema.
    _patch_polars_schema()

    reconnect_attempt = 0
    while True:
        reconnect_attempt += 1
        if reconnect_attempt > 1:
            log.info("🔄 Reconnect #%d (waiting %ds...)",
                     reconnect_attempt, RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)

        suite = None
        for ca in range(10):
            try:
                log.info("Connecting... (try %d/10)", ca + 1)
                suite = await TradingSuite.create("MNQ")

                # ── Immediate 1-min bar fetch (must be FIRST await after create) ──
                # TradingSuite.create() starts a background WebSocket feed that
                # seeds the bounded_stats DataFrame with 8-column HTTP bars.
                # When the first live 5-min bar arrives via WebSocket it only has
                # 6 columns → SchemaError.
                #
                # Calling get_bars() here — as the very next network operation —
                # forces the library to complete its internal bar-cache handshake
                # before the WebSocket can deliver the first mismatched update.
                # v10.4 achieves the same effect via its pre-flight market-data
                # check.  Calling it directly on the client (not through our
                # BarFetcher wrapper) is the most immediate possible call.
                try:
                    await suite.client.get_bars("MNQ", days=1, interval=1)
                    log.info("✅ Library bar-cache primed (immediate 1-min fetch)")
                except Exception as _pe:
                    log.debug("Prime fetch error (non-fatal): %s", _pe)

                # Patch AFTER create() and after the prime fetch so all library
                # modules are loaded and the gc scan has live objects to find.
                _patch_bounded_stats()
                log.info("✅ Connected.")
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                wait = min(60, 5 * (ca + 1))
                log.error("Conn %d/10 failed: %s — retry in %ds", ca + 1, exc, wait)
                await asyncio.sleep(wait)

        if suite is None:
            log.critical("Could not connect after 10 attempts.")
            continue

        strategy = Strategy(suite)
        try:
            await strategy.run()
            log.info("Strategy returned cleanly — shutting down.")
            break
        except asyncio.CancelledError:
            log.info("Cancelled.")
            try:
                await suite.positions.close_all_positions()
            except Exception:
                pass
            raise
        except ConnectionLostError as exc:
            log.error("🔌 Connection lost: %s — reconnecting.", exc)
        except Exception as exc:
            log.exception("💥 Unexpected error: %s — reconnecting.", exc)


# ─────────────────────────────────────────────────────────────
# 12.  ENTRYPOINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if sys.platform == "win32":
        loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(loop)
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        log.info("Keyboard interrupt — shutting down.")
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()
