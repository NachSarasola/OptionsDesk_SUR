"""Tests del módulo market_intel (mockeados: no hacen requests reales)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from optionsdesk.data.providers.market_intel import (
    MarketIntel,
    _rating_label,
    _yahoo_analyst,
    _stocktwits_social,
    fetch_market_intel,
)


def _mock_yahoo_response() -> dict:
    return {"quoteSummary": {"result": [{
        "financialData": {
            "recommendationMean": {"raw": 2.1},
            "targetMeanPrice": {"raw": 35.0},
            "currentPrice": {"raw": 30.0},
        },
        "recommendationTrend": {"trend": [{"strongBuy": 3, "buy": 4, "hold": 2, "sell": 1, "strongSell": 0}]},
    }], "error": None}}


def _mock_stocktwits_response() -> dict:
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    msgs = []
    for i in range(10):
        ts = (now - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sentiment = {"basic": "Bullish"} if i < 6 else {"basic": "Bearish"}
        msgs.append({"id": i, "created_at": ts, "entities": {"sentiment": sentiment}})
    return {"messages": msgs}


def test_rating_label():
    assert _rating_label(1.2) == "Compra fuerte"
    assert _rating_label(2.0) == "Compra"
    assert _rating_label(2.5) == "Neutral"
    assert _rating_label(3.5) == "Venta"
    assert _rating_label(4.8) == "Venta fuerte"


def test_yahoo_analyst_parses_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_yahoo_response()
    with patch("optionsdesk.data.providers.market_intel._requests") as mock_req:
        mock_req.get.return_value = mock_resp
        rating, upside, n = _yahoo_analyst("GGAL")
    assert rating == 2.1
    assert upside == pytest.approx((35.0 / 30.0 - 1.0) * 100.0, rel=0.01)
    assert n == 10


def test_stocktwits_counts_recent_and_sentiment():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_stocktwits_response()
    with patch("optionsdesk.data.providers.market_intel._requests") as mock_req:
        mock_req.get.return_value = mock_resp
        msgs, bull, bear = _stocktwits_social("GGAL")
    assert msgs is not None and msgs > 0
    assert bull is not None and bull > 50.0    # 6/10 = 60% bullish


def test_market_intel_score_delta_buy():
    intel = MarketIntel(
        ticker="GGAL", analyst_rating=1.8, analyst_label="Compra",
        target_upside_pct=25.0, n_analysts=8,
        social_msgs_24h=40, social_bull_pct=70.0, social_bear_pct=30.0,
    )
    assert intel.score_delta() > 0   # consenso compra + upside + alcista


def test_market_intel_score_delta_sell():
    intel = MarketIntel(
        ticker="GGAL", analyst_rating=4.5, analyst_label="Venta fuerte",
        target_upside_pct=-15.0, n_analysts=5,
        social_msgs_24h=20, social_bull_pct=30.0, social_bear_pct=70.0,
    )
    assert intel.score_delta() < 0


def test_market_intel_score_delta_capped():
    intel = MarketIntel(
        ticker="GGAL", analyst_rating=1.0, analyst_label="Compra fuerte",
        target_upside_pct=50.0, n_analysts=15,
        social_msgs_24h=200, social_bull_pct=90.0, social_bear_pct=10.0,
    )
    assert intel.score_delta() <= 10.0   # cap en +10


def test_fetch_market_intel_uses_cache(tmp_path):
    """Si hay cache válido, no hace requests."""
    import time
    from optionsdesk.data.providers.market_intel import _cache_path, _write_cache
    analyst_p = _cache_path("GGAL.US", "analyst", tmp_path)
    _write_cache(analyst_p, {"rating": 2.0, "upside": 10.0, "n_analysts": 5})
    social_p = _cache_path("GGAL.US", "social", tmp_path)
    _write_cache(social_p, {"msgs": 30, "bull_pct": 60.0, "bear_pct": 40.0})

    with patch("optionsdesk.data.providers.market_intel._requests") as mock_req:
        with patch("optionsdesk.config.settings.settings") as mock_s:
            mock_s.adr_map.return_value = {"GGAL": ("GGAL.US", 10.0)}
            mock_s.history_dir = tmp_path.parent
            intel = fetch_market_intel("GGAL", cache_dir=tmp_path)

        mock_req.get.assert_not_called()   # no hizo requests
    assert intel is not None
    assert intel.analyst_rating == 2.0
    assert intel.social_msgs_24h == 30


import pytest
