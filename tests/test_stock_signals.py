from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from optionsdesk.config.costs import CostModel
from optionsdesk.data.providers.base import Quote
from optionsdesk.signals.stock_signals import scan_stock_symbol


BA = ZoneInfo("America/Argentina/Buenos_Aires")

# Comision realista de un ALyC AR para trader activo (~0.2%). iva_rate es FRACCION.
LOW_COSTS = CostModel(
    stock_commission_pct=0.2,
    option_commission_pct=0.2,
    stock_market_fee_pct=0.08,
    option_market_fee_pct=0.08,
    exercise_fee_pct=0.0,
    exercise_market_fee_pct=0.0,
    iva_rate=0.21,
)

# Comision punitiva (3%): el ida-y-vuelta se come el edge de un swing 2R.
HIGH_COSTS = CostModel(
    stock_commission_pct=3.0,
    option_commission_pct=3.0,
    stock_market_fee_pct=0.08,
    option_market_fee_pct=0.08,
    exercise_fee_pct=0.0,
    exercise_market_fee_pct=0.0,
    iva_rate=0.21,
)


def _quote(symbol: str, bid: float, ask: float, last: float, ts: datetime) -> Quote:
    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=10_000.0,
        timestamp=ts,
        source="TEST",
        received_at=ts,
        book_ts=ts,
    )


def _daily(closes: list[float], vols: list[float] | None = None) -> pd.DataFrame:
    start = datetime(2026, 1, 2)
    rows = []
    for i, c in enumerate(closes):
        rows.append({
            "date": start + timedelta(days=i),
            "open": c - 0.3,
            "high": c + 0.5,
            "low": c - 0.5,
            "close": c,
            "volume": (vols[i] if vols else 1000.0),
        })
    return pd.DataFrame(rows)


def _uptrend_then_consolidation() -> list[float]:
    """Tendencia alcista limpia de 45 ruedas + 15 de consolidacion (RSI sano)."""
    ramp = [100.0 + i * 1.1 for i in range(45)]            # 100 -> ~148
    base = ramp[-1]
    consol = [base + (1.5 if i % 2 else 0.5) for i in range(15)]  # oscila ~148.5-150
    return ramp + consol


def test_swing_breakout_generates_signal_net_of_costs():
    closes = _uptrend_then_consolidation()
    vols = [1000.0] * (len(closes) - 1) + [2000.0]        # volumen de ruptura
    daily = _daily(closes, vols)
    prior_high = max(c + 0.5 for c in closes[-21:-1])
    entry = prior_high * 1.01                               # rompe el pivote, no extendido
    now = datetime(2026, 6, 3, 12, 0, tzinfo=BA)
    quote = _quote("GGAL", entry - 0.2, entry, entry, now)

    signals = scan_stock_symbol("GGAL", quote, daily=daily, now=now, costs=LOW_COSTS)

    assert signals
    sig = signals[0]
    assert sig.strategy == "SWING"
    assert sig.signal_type == "SWING_BREAKOUT"
    assert sig.stop_price < sig.entry_price < sig.target_price
    assert sig.net_rr >= 0.6          # deja edge despues de costos
    assert sig.cost_pct > 0.0         # el costo quedo expuesto


def test_high_costs_filter_marginal_swing():
    """Con comision punitiva (3%) el ida-y-vuelta se come el edge -> sin senal."""
    closes = _uptrend_then_consolidation()
    vols = [1000.0] * (len(closes) - 1) + [2000.0]
    daily = _daily(closes, vols)
    prior_high = max(c + 0.5 for c in closes[-21:-1])
    entry = prior_high * 1.01
    now = datetime(2026, 6, 3, 12, 0, tzinfo=BA)
    quote = _quote("GGAL", entry - 0.2, entry, entry, now)

    signals = scan_stock_symbol("GGAL", quote, daily=daily, now=now, costs=HIGH_COSTS)

    assert signals == []


def test_does_not_chase_extended_breakout():
    """Entrada >5% sobre el pivote = tarde, no se emite."""
    closes = _uptrend_then_consolidation()
    daily = _daily(closes)
    prior_high = max(c + 0.5 for c in closes[-21:-1])
    entry = prior_high * 1.08                               # ya corrio 8% sobre el pivote
    now = datetime(2026, 6, 3, 12, 0, tzinfo=BA)
    quote = _quote("GGAL", entry - 0.2, entry, entry, now)

    signals = scan_stock_symbol("GGAL", quote, daily=daily, now=now, costs=LOW_COSTS)

    assert all(s.signal_type != "SWING_BREAKOUT" for s in signals)


def _swing_sig(signal_type: str) -> "object":
    from optionsdesk.signals.stock_signals import StockSignal
    return StockSignal(
        symbol="GGAL", strategy="SWING", signal_type=signal_type,
        created_at=datetime(2026, 6, 3, 12, 0, tzinfo=BA),
        entry_price=100.0, stop_price=95.0, target_price=110.0,
        score=60.0, rationale="base",
    )


def test_smc_blocks_long_against_bearish_mm_cycle():
    from optionsdesk.signals.smc import SmcContext, MmRead
    from optionsdesk.signals.stock_signals import _smc_quality
    ctx = SmcContext(mm=MmRead(
        ema5=1, ema13=2, ema50=3, ema200=4, stack="bearish",
        level_count=0, peak_formation=False, phase="x", ema13_cross_50="below",
    ))
    delta, block, notes = _smc_quality(_swing_sig("SWING_BREAKOUT"), ctx, spot=100.0)
    assert block is not None       # no comprar contra el ciclo bajista


def test_smc_boosts_pullback_in_discount():
    from optionsdesk.signals.smc import SmcContext, PdaArray
    from optionsdesk.signals.stock_signals import _smc_quality
    pda = PdaArray(
        swing_high=120.0, swing_low=80.0, equilibrium=100.0,
        premium_band=104.7, discount_band=95.3, zone="discount",
        ote_bullish_lo=104.7, ote_bullish_hi=111.4,
        ote_bearish_lo=111.4, ote_bearish_hi=115.4,
    )
    ctx = SmcContext(pda=pda)
    delta, block, notes = _smc_quality(_swing_sig("SWING_TREND_PULLBACK"), ctx, spot=90.0)
    assert block is None
    assert delta > 0
    assert any("descuento" in n for n in notes)


def test_smc_penalizes_pullback_in_premium():
    from optionsdesk.signals.smc import SmcContext, PdaArray
    from optionsdesk.signals.stock_signals import _smc_quality
    pda = PdaArray(
        swing_high=120.0, swing_low=80.0, equilibrium=100.0,
        premium_band=104.7, discount_band=95.3, zone="premium",
        ote_bullish_lo=104.7, ote_bullish_hi=111.4,
        ote_bearish_lo=111.4, ote_bearish_hi=115.4,
    )
    ctx = SmcContext(pda=pda)
    delta, block, notes = _smc_quality(_swing_sig("SWING_TREND_PULLBACK"), ctx, spot=115.0)
    assert block is None
    assert delta < 0               # comprar pullback en premium = caro


def test_smc_blocks_breakout_into_unswept_bsl():
    from optionsdesk.signals.smc import SmcContext, LiquidityLevel
    from optionsdesk.signals.stock_signals import _smc_quality
    # BSL sin barrer a +1% arriba: comprás directo contra resistencia, sin recorrido.
    ctx = SmcContext(liquidity=[LiquidityLevel("BSL", 101.0, "PWH", False, "weekly")])
    delta, block, notes = _smc_quality(_swing_sig("SWING_BREAKOUT"), ctx, spot=100.0)
    assert block is not None


def test_no_daily_history_returns_empty():
    now = datetime(2026, 6, 3, 12, 0, tzinfo=BA)
    quote = _quote("GGAL", 99.0, 100.0, 99.5, now)
    assert scan_stock_symbol("GGAL", quote, daily=None, now=now) == []


def test_requires_valid_bid_ask_book():
    now = datetime(2026, 6, 3, 12, 0, tzinfo=BA)
    daily = _daily(_uptrend_then_consolidation())
    quote = _quote("GGAL", 0.0, 100.0, 99.5, now)
    assert scan_stock_symbol("GGAL", quote, daily=daily, now=now, costs=LOW_COSTS) == []


# ── Tests: signal_grade ───────────────────────────────────────────────────────

def test_signal_grade_s():
    from optionsdesk.signals.stock_signals import signal_grade
    assert signal_grade(95.0) == "S"
    assert signal_grade(88.0) == "S"

def test_signal_grade_a():
    from optionsdesk.signals.stock_signals import signal_grade
    assert signal_grade(87.9) == "A"
    assert signal_grade(78.0) == "A"

def test_signal_grade_b():
    from optionsdesk.signals.stock_signals import signal_grade
    assert signal_grade(77.9) == "B"
    assert signal_grade(66.0) == "B"

def test_signal_grade_c():
    from optionsdesk.signals.stock_signals import signal_grade
    assert signal_grade(65.9) == "C"
    assert signal_grade(52.0) == "C"

def test_signal_grade_d():
    from optionsdesk.signals.stock_signals import signal_grade
    assert signal_grade(51.9) == "D"
    assert signal_grade(0.0) == "D"

def test_signal_grade_monotone():
    """Grade es monotona: scores mas altos nunca dan grade peor."""
    from optionsdesk.signals.stock_signals import signal_grade
    _ORDER = ["D", "C", "B", "A", "S"]
    prev_idx = 0
    for score in [10, 30, 52, 66, 78, 88, 100]:
        grade = signal_grade(score)
        idx = _ORDER.index(grade)
        assert idx >= prev_idx, f"score={score} grade={grade} regresa respecto a prev"
        prev_idx = idx


# ── Tests: SMC_REVERSAL setup ─────────────────────────────────────────────────

def test_smc_reversal_not_generated_without_sweeps():
    """Sin sweeps en el SMC context no se genera SMC_REVERSAL."""
    from optionsdesk.signals.stock_signals import _scan_smc_reversal
    import pandas as pd
    from datetime import datetime
    daily = pd.DataFrame({
        "date":   pd.date_range("2024-01-02", periods=30, freq="B"),
        "open":   [1000.0] * 30, "high": [1010.0] * 30,
        "low":    [990.0]  * 30, "close": [1005.0] * 30,
        "volume": [1e6]    * 30,
    })
    # Contexto SMC sin sweeps
    class FakeSmc:
        sweeps = []
        liquidity = []
        ote = None
        pda = None
        mm = None
        bos = None
    quote = _quote("GGAL", 1005, 1007, 1005, datetime.now())
    result = _scan_smc_reversal("GGAL", quote, daily, None, datetime.now(), FakeSmc())
    assert result == []

def test_smc_reversal_requires_reversed_sweep():
    """Un sweep sin reversión (el precio no cerró de vuelta) no activa el setup."""
    from optionsdesk.signals.stock_signals import _scan_smc_reversal
    import pandas as pd
    from datetime import datetime

    class FakeLv:
        kind = "SSL"; price = 990.0; label = "PDL"; swept = False; timeframe = "daily"
    class FakeSweep:
        direction = "down"; reversed = False; bar_index = 25
        level = FakeLv()
    class FakeSmc:
        sweeps = [FakeSweep()]
        liquidity = [FakeLv()]
        ote = None; pda = None; mm = None; bos = None

    daily = pd.DataFrame({
        "date":   pd.date_range("2024-01-02", periods=30, freq="B"),
        "open":   [1000.0]*30, "high": [1010.0]*30,
        "low":    [990.0]*30,  "close": [1005.0]*30,
        "volume": [1e6]*30,
    })
    quote = _quote("GGAL", 1005, 1007, 1005, datetime.now())
    result = _scan_smc_reversal("GGAL", quote, daily, None, datetime.now(), FakeSmc())
    assert result == []
