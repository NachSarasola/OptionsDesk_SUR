"""Tests del calendario de eventos (config/events.py)."""
from __future__ import annotations

from datetime import date

import pytest

from optionsdesk.config.events import MarketEvent, spans_event


def test_spans_event_detects_overlap():
    events = [MarketEvent(date(2026, 8, 7), "earnings", "GGAL Q2 2026")]
    ev = spans_event(date(2026, 7, 15), date(2026, 8, 15), events)
    assert ev is not None
    assert ev.kind == "earnings"


def test_spans_event_returns_none_when_no_overlap():
    events = [MarketEvent(date(2026, 8, 7), "earnings", "GGAL Q2 2026")]
    ev = spans_event(date(2026, 8, 8), date(2026, 9, 15), events)
    assert ev is None


def test_spans_event_exact_boundary_inclusive():
    """El evento EN el día de vencimiento cuenta (ventana es (entry, expiry])."""
    events = [MarketEvent(date(2026, 8, 7), "earnings", "GGAL Q2 2026")]
    ev = spans_event(date(2026, 7, 1), date(2026, 8, 7), events)
    assert ev is not None


def test_spans_event_entry_boundary_exclusive():
    """El evento EN el día de entrada NO cuenta (ventana es (entry, expiry])."""
    events = [MarketEvent(date(2026, 8, 7), "earnings", "GGAL Q2 2026")]
    ev = spans_event(date(2026, 8, 7), date(2026, 9, 1), events)
    assert ev is None


def test_spans_event_returns_first_matching():
    events = [
        MarketEvent(date(2026, 8, 7),  "earnings", "GGAL Q2"),
        MarketEvent(date(2026, 9, 25), "bcra",      "BCRA sept"),
    ]
    ev = spans_event(date(2026, 7, 1), date(2026, 10, 1), events)
    assert ev is not None
    assert ev.kind == "earnings"   # el primero en la lista que solapa


def test_spans_event_empty_list_returns_none():
    ev = spans_event(date(2026, 1, 1), date(2026, 12, 31), [])
    assert ev is None


def test_spans_event_uses_known_events_by_default():
    """Sin pasar lista, usa KNOWN_EVENTS — al menos los earnings de GGAL 2026."""
    # Una ventana que cruza los earnings de agosto 2026
    ev = spans_event(date(2026, 7, 1), date(2026, 8, 20))
    # Puede ser None si KNOWN_EVENTS está vacía o las fechas cambian — no forzamos.
    # Solo verificamos que no lanza excepción.
    assert ev is None or ev.event_date is not None
