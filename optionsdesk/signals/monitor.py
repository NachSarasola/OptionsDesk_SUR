"""Monitor intradía de posiciones abiertas.

Lee data/open_positions.jsonl, recomputa el porcentaje de captura actual
de cada posición usando CRR, y dispara una alerta de Telegram cuando se
alcanza el objetivo de salida definido por el horizon optimizer.

Formato de open_positions.jsonl (una posición por línea JSON):
{
  "symbol":             "GGALC9000F",
  "strategy":           "COVERED_CALL",
  "strike":             9000.0,
  "spot_entry":         8500.0,
  "premium_received":   300.0,
  "net_outlay":         8700.0,
  "iv_entry":           0.55,
  "days_entry":         30,
  "entry_date":         "2026-05-21",
  "target_exit_days":   15,
  "target_capture_pct": 75.0,
  "caucion_tna":        60.0
}

Uso como script (hook del recorder):
    python -m optionsdesk.signals.monitor
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from optionsdesk.core.pricing import crr_price

logger = logging.getLogger(__name__)

_DEFAULT_POSITIONS_FILE = Path("data/open_positions.jsonl")
_CRR_STEPS = 50


@dataclass
class OpenPosition:
    symbol: str
    strategy: str
    strike: float
    spot_entry: float
    premium_received: float
    net_outlay: float
    iv_entry: float
    days_entry: int
    entry_date: date
    target_exit_days: int
    target_capture_pct: float
    caucion_tna: float = 60.0
    # Parámetros de gestión activa (v2.4) — retrocompatibles via .get() con defaults
    max_loss_mult: float = 2.0   # stop a max_loss_mult × prima cobrada
    roll_dte: int = 21           # señal de roll cuando quedan ≤ roll_dte días
    defend_delta: float = 0.50   # señal de defensa cuando |delta| ≥ defend_delta
    # Cantidad de contratos abiertos (v3.1) — default 1 para retrocompatibilidad
    contracts: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "OpenPosition":
        return cls(
            symbol=d["symbol"],
            strategy=d["strategy"],
            strike=float(d["strike"]),
            spot_entry=float(d["spot_entry"]),
            premium_received=float(d["premium_received"]),
            net_outlay=float(d["net_outlay"]),
            iv_entry=float(d["iv_entry"]),
            days_entry=int(d["days_entry"]),
            entry_date=date.fromisoformat(d["entry_date"]),
            target_exit_days=int(d["target_exit_days"]),
            target_capture_pct=float(d["target_capture_pct"]),
            caucion_tna=float(d.get("caucion_tna", 60.0)),
            max_loss_mult=float(d.get("max_loss_mult", 2.0)),
            roll_dte=int(d.get("roll_dte", 21)),
            defend_delta=float(d.get("defend_delta", 0.50)),
            contracts=int(d.get("contracts", 1)),
        )

    @property
    def days_held(self) -> int:
        return (date.today() - self.entry_date).days

    @property
    def days_remaining(self) -> int:
        return max(self.days_entry - self.days_held, 0)

    @property
    def opt_type(self) -> str:
        return "C" if "CALL" in self.strategy else "P"


class PositionMonitor:
    """Monitorea posiciones abiertas y dispara alertas al alcanzar el objetivo."""

    def __init__(
        self,
        positions_file: Path = _DEFAULT_POSITIONS_FILE,
        alerter=None,
    ) -> None:
        self._file     = positions_file
        # Import perezoso para evitar ciclo (alerts → monitor → alerts).
        self._alerter  = alerter

    def _get_alerter(self):
        if self._alerter is None:
            from optionsdesk.signals.alerts import TelegramAlerter
            self._alerter = TelegramAlerter()
        return self._alerter

    def load_positions(self) -> list[OpenPosition]:
        if not self._file.exists():
            return []
        positions: list[OpenPosition] = []
        with self._file.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    positions.append(OpenPosition.from_dict(json.loads(line)))
                except (KeyError, ValueError) as exc:
                    logger.warning("Línea inválida en posiciones: %.60s — %s", line, exc)
        return positions

    def compute_capture(
        self,
        pos: OpenPosition,
        current_spot: Optional[float] = None,
    ) -> float:
        """Porcentaje de la ganancia total ya capturado al momento actual.

        Si no se pasa current_spot usa el spot de entrada (caso sin datos en vivo).
        """
        S     = current_spot or pos.spot_entry
        T_rem = pos.days_remaining / 365.0
        r     = math.log(1.0 + pos.caucion_tna / 100.0)

        if T_rem < 1.0 / 365.0 or pos.iv_entry <= 0:
            current_ov = (
                max(S - pos.strike, 0.0)
                if pos.opt_type == "C"
                else max(pos.strike - S, 0.0)
            )
        else:
            current_ov = crr_price(
                S, pos.strike, T_rem, r, pos.iv_entry, pos.opt_type, steps=_CRR_STEPS
            )

        is_long = "LONG" in pos.strategy
        if is_long:
            pnl_now = current_ov - pos.net_outlay
            return (pnl_now / pos.net_outlay * 100.0) if pos.net_outlay > 0 else 0.0
        else:
            pnl_now = pos.premium_received - current_ov
            pnl_full = pos.premium_received
            if pnl_full <= 0:
                return 0.0
            return max(pnl_now / pnl_full * 100.0, 0.0)

    def check_and_alert(
        self,
        positions: Optional[list[OpenPosition]] = None,
        current_spot: Optional[float] = None,
        current_iv: Optional[float] = None,
    ) -> list[OpenPosition]:
        """Evalúa cada posición y alerta según la señal de gestión activa.

        Devuelve la lista de posiciones con señal accionable (no HOLD).
        """
        from optionsdesk.signals.management import evaluate_position, SignalType

        if positions is None:
            positions = self.load_positions()
        triggered: list[OpenPosition] = []
        alerter = self._get_alerter()

        for pos in positions:
            spot = current_spot or pos.spot_entry
            signal = evaluate_position(pos, spot, current_iv)

            logger.debug(
                "%s  captura=%.1f%%  señal=%s",
                pos.symbol, signal.capture_pct, signal.signal_type,
            )

            if signal.signal_type == SignalType.HOLD:
                continue

            alerter.send_management_signal(signal, pos)
            triggered.append(pos)

        return triggered

    def run_once(self, current_spot: Optional[float] = None) -> None:
        """Carga posiciones y dispara alertas si corresponde. Sin side effects extras."""
        positions = self.load_positions()
        if not positions:
            return
        self.check_and_alert(positions, current_spot)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    spot_arg: Optional[float] = float(sys.argv[1]) if len(sys.argv) > 1 else None
    PositionMonitor().run_once(spot_arg)
