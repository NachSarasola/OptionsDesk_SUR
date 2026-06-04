import pytest
from datetime import datetime, timedelta
from optionsdesk.core.greeks_engine import compute_chain_greeks
from optionsdesk.data.providers.base import OptionsChain, Quote

@pytest.fixture
def mock_chain():
    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, datetime.now())
    options = {
        "GFGC4000A": Quote("GFGC4000A", 150.0, 160.0, 155.0, 100, datetime.now()),
        "GFGV4000A": Quote("GFGV4000A", 140.0, 150.0, 145.0, 100, datetime.now()),
        "GFGC4200A": Quote("GFGC4200A", 50.0, 60.0, 55.0, 10, datetime.now()), # OTM call
    }
    return OptionsChain("GGAL", spot_quote, options)

@pytest.fixture
def mock_expiry_cal():
    from datetime import date, timedelta
    return {"A": date.today() + timedelta(days=30)}

def test_chain_greeks_basic(mock_chain, mock_expiry_cal):
    greeks = compute_chain_greeks(mock_chain, mock_expiry_cal)
    
    assert len(greeks) == 3
    
    c4000 = greeks["GFGC4000A"]
    assert c4000.option_type == "C"
    assert c4000.strike == 4000.0
    assert c4000.moneyness == "ATM"
    assert c4000.delta > 0.45 and c4000.delta < 0.65  # ATM Call delta should be around 0.5
    assert c4000.gamma > 0
    assert c4000.theta < 0
    assert c4000.vega > 0

    p4000 = greeks["GFGV4000A"]
    assert p4000.option_type == "P"
    assert p4000.moneyness == "ATM"
    assert p4000.delta < -0.35 and p4000.delta > -0.65 # ATM Put delta should be around -0.5
    
    c4200 = greeks["GFGC4200A"]
    assert c4200.moneyness == "OTM"


def test_chain_greeks_marks_stale_as_not_tradeable(mock_expiry_cal):
    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, datetime.now())
    old_ts = datetime.now() - timedelta(hours=1)
    options = {
        "GFGC4000A": Quote("GFGC4000A", 150.0, 160.0, 155.0, 100, old_ts),
    }
    chain = OptionsChain("GGAL", spot_quote, options)
    greeks = compute_chain_greeks(chain, mock_expiry_cal)
    g = greeks["GFGC4000A"]
    assert g.is_tradeable is False
    assert "stale" in g.liquidity_reason


def test_chain_greeks_quote_age_can_use_replay_clock(mock_expiry_cal):
    replay_now = datetime(2026, 6, 2, 12, 0)
    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, replay_now)
    options = {
        "GFGC4000A": Quote("GFGC4000A", 150.0, 160.0, 155.0, 100, replay_now),
    }
    chain = OptionsChain("GGAL", spot_quote, options)

    greeks = compute_chain_greeks(chain, mock_expiry_cal, now=replay_now + timedelta(seconds=10))

    assert greeks["GFGC4000A"].quote_age_s == pytest.approx(10.0)
    assert greeks["GFGC4000A"].is_tradeable is True


def test_chain_greeks_uses_observed_expiry_calendar_from_chain():
    from datetime import date, timedelta

    spot_quote = Quote("GGAL", 4000.0, 4000.0, 4000.0, 1000, datetime.now())
    options = {
        "GFGC4000JU": Quote("GFGC4000JU", 150.0, 160.0, 155.0, 100, datetime.now()),
    }
    chain = OptionsChain(
        "GGAL",
        spot_quote,
        options,
        expiry_calendar={"JU": date.today() + timedelta(days=30)},
    )

    greeks = compute_chain_greeks(chain, expiry_calendar={})

    assert "GFGC4000JU" in greeks
