"""Calendario de eventos de mercado relevantes para GGAL.

Mantener actualizado a mano cada ciclo trimestral. No hay API gratuita
para eventos corporativos argentinos — la fuente es el sitio de GGAL y el
calendario de BCRA.

Uso en el recommender: una opción cuyo vencimiento cruza un evento conocido
recibe una penalización de score y un warning — mayor incertidumbre (gap risk
+ potencial IV crush post-evento).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass
class MarketEvent:
    event_date: date
    kind: str          # "earnings" | "bcra" | "election" | "macro"
    description: str


# ── Lista de eventos conocidos — actualizar cada ciclo ────────────────────────

KNOWN_EVENTS: list[MarketEvent] = [
    # Resultados trimestrales de GGAL (fechas aproximadas — verificar en CNV/BYMA)
    MarketEvent(date(2026, 8, 7),  "earnings", "GGAL Resultados Q2 2026"),
    MarketEvent(date(2026, 11, 5), "earnings", "GGAL Resultados Q3 2026"),
    MarketEvent(date(2027, 2, 26), "earnings", "GGAL Resultados Q4 2026"),
    # Reuniones BCRA (último jueves del mes aproximadamente — verificar calendario)
    # Agregar fechas confirmadas cuando estén disponibles.
]


# ── API pública ───────────────────────────────────────────────────────────────

def spans_event(
    entry_date: date,
    expiration: date,
    events: Optional[list[MarketEvent]] = None,
) -> Optional[MarketEvent]:
    """Devuelve el primer evento conocido que cae dentro de (entry_date, expiration].

    None si no hay eventos en la ventana.
    """
    ev_list = events if events is not None else KNOWN_EVENTS
    for ev in ev_list:
        if entry_date < ev.event_date <= expiration:
            return ev
    return None
