"""Tests del arranque explicito del recorder."""
from __future__ import annotations

import pytest

from optionsdesk.data import recorder


def test_build_provider_requires_real_credentials(monkeypatch):
    monkeypatch.setattr(recorder.settings, "iol_user", "")
    monkeypatch.setattr(recorder.settings, "iol_password", "")
    monkeypatch.setattr(recorder.settings, "hb_dni", "")
    monkeypatch.setattr(recorder.settings, "hb_user", "")
    monkeypatch.setattr(recorder.settings, "hb_password", "")
    monkeypatch.setattr(recorder.settings, "hb_broker_id", 0)

    with pytest.raises(RuntimeError, match="Configura credenciales"):
        recorder._build_provider()
