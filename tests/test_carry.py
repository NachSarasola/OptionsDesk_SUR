from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from optionsdesk.core.carry import estimate_carry
from optionsdesk.data.providers.base import OptionsChain, Quote


def test_estimate_carry_uses_fresh_put_call_parity():
    now = datetime.now()
    expiry = date.today() + timedelta(days=30)
    spot = Quote("GGAL", 3995.0, 4005.0, 4000.0, 100_000.0, now)
    chain = OptionsChain(
        "GGAL",
        spot,
        {
            "GFGC4000T": Quote("GFGC4000T", 228.0, 232.0, 230.0, 100.0, now),
            "GFGV4000T": Quote("GFGV4000T", 98.0, 102.0, 100.0, 100.0, now),
        },
    )

    carry = estimate_carry(chain, {"T": expiry}, fallback_tna_pct=45.0)

    assert carry.source == "put_call_parity"
    assert carry.pairs_used == 1
    assert carry.tna_pct > 20.0
    assert carry.continuous_rate > 0.0


def test_estimate_carry_fallback_is_explicit_when_pairs_are_missing():
    now = datetime.now()
    chain = OptionsChain(
        "GGAL",
        Quote("GGAL", 3995.0, 4005.0, 4000.0, 100_000.0, now),
        {"GFGC4000T": Quote("GFGC4000T", 228.0, 232.0, 230.0, 100.0, now)},
    )

    carry = estimate_carry(chain, {"T": date.today() + timedelta(days=30)}, fallback_tna_pct=45.0)

    assert carry.source == "configured_caucion_fallback"
    assert carry.tna_pct == pytest.approx(45.0)

