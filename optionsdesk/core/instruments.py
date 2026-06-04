from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class OptionContract:
    """Un contrato de opción de GGAL en BYMA (lote = 100 acciones)."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    expiry_code: str  # código raw del ticker (para agrupar por vencimiento)

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def days_to_expiry(self) -> int:
        return max((self.expiration - date.today()).days, 0)

    @property
    def years_to_expiry(self) -> float:
        return self.days_to_expiry / 365.0

    def moneyness(self, spot: float, atm_threshold_pct: float = 2.0) -> str:
        """ITM / ATM / OTM desde la perspectiva del comprador."""
        if spot <= 0:
            return "N/A"
        rel = abs(self.strike - spot) / spot * 100
        if rel <= atm_threshold_pct:
            return "ATM"
        if self.is_call:
            return "ITM" if self.strike < spot else "OTM"
        return "ITM" if self.strike > spot else "OTM"


# Formato de ticker BYMA para GGAL:
# GFG + tipo (C=call, V=put) + strike numérico + código de vencimiento (letras)
# Ejemplo: GFGC7000F → call, strike 7000, vencimiento código F
# Nota: la convención puede variar; validar contra datos reales de pyhomebroker.
_TICKER_RE = re.compile(
    r"^GFG(?P<otype>[CVcv])(?P<strike>\d+(?:[.,]\d+)?)(?P<code>[A-Z]+)$"
)


def parse_option_symbol(
    symbol: str,
    expiry_calendar: dict[str, date],
    spot_hint: Optional[float] = None,
) -> Optional[OptionContract]:
    """Parseador genérico de opciones.

    Extrae underlying, call/put, strike y expiration.
    Si se provee spot_hint, corrige strikes desfasados de BYMA (multiplicados x10 o x100).
    """
    m = _TICKER_RE.match(symbol.strip().upper())
    if not m:
        return None

    # Usamos default GGAL para opciones
    otype = OptionType.CALL if m.group("otype").upper() == "C" else OptionType.PUT
    strike = float(m.group("strike").replace(",", "."))

    # Corrección BYMA usando el spot
    if spot_hint is not None and spot_hint > 0:
        while strike > spot_hint * 3:
            strike /= 10.0
        while strike < spot_hint / 3:
            strike *= 10.0

    code = m.group("code")
    expiration = expiry_calendar.get(code)
    if expiration is None:
        return None

    return OptionContract(
        symbol=symbol.upper(),
        underlying="GGAL",
        option_type=otype,
        strike=strike,
        expiration=expiration,
        expiry_code=code,
    )


def days_to_expiry(symbol: str, expiry_calendar: Optional[dict[str, date]] = None) -> int:
    """Días al vencimiento dado un símbolo de opción BYMA.

    Usa el expiry_calendar si se provee; de lo contrario intenta cargarlo
    de instruments.yaml. Devuelve 0 si no puede determinar la fecha.
    """
    cal = expiry_calendar
    if cal is None:
        try:
            cal = load_expiry_calendar()
        except Exception as e:
            logger.debug("load_expiry_calendar fallo en days_to_expiry: %s", e)
            return 0
    contract = parse_option_symbol(symbol, cal)
    return contract.days_to_expiry if contract is not None else 0


def merge_expiry_calendars(
    fallback: Optional[dict[str, date]] = None,
    observed: Optional[dict[str, date]] = None,
) -> dict[str, date]:
    """Combina calendario local y observado; el feed live tiene prioridad."""
    merged = dict(fallback or {})
    merged.update(observed or {})
    return merged


def load_expiry_calendar(config_path: Optional[str] = None) -> dict[str, date]:
    """Carga el mapa código → fecha desde instruments.yaml."""
    import yaml
    from pathlib import Path

    path = Path(config_path or "optionsdesk/config/instruments.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        code: date.fromisoformat(str(dt_str))
        for code, dt_str in (cfg.get("expiry_calendar") or {}).items()
    }
