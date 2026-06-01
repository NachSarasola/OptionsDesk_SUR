from __future__ import annotations

import os

import pytest

from optionsdesk.data.providers.iol import IOLProvider


@pytest.mark.integration
def test_iol_live_read_only_smoke():
    if not (os.environ.get("IOL_USER") and os.environ.get("IOL_PASSWORD")):
        pytest.skip("IOL credentials not configured")

    p = IOLProvider()
    p.connect()
    try:
        spot = p.get_spot()
        assert spot is not None
        chain = p.get_options_chain()
        assert chain is not None
        health = p.get_health()
        assert health.source == "IOL"
    finally:
        p.disconnect()
