"""Alertas via Telegram Bot API.

Envía avisos cuando una oportunidad supera el umbral de spread configurado.
Obtené el token del @BotFather y el chat_id con @userinfobot.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import requests

from optionsdesk.config.settings import settings
from optionsdesk.core.rates import RateResult

logger = logging.getLogger(__name__)

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlerter:
    """Envía alertas de oportunidades a un chat de Telegram."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token = token or settings.telegram_token
        self._chat_id = chat_id or settings.telegram_chat_id

    def _send(self, text: str) -> bool:
        if not self._token or not self._chat_id:
            logger.warning("Telegram no configurado; alerta omitida.")
            return False
        url = _API_URL.format(token=self._token)
        try:
            r = requests.post(
                url,
                json={"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("Fallo al enviar alerta Telegram: %s", e)
            return False

    def send_opportunity(self, result: RateResult) -> bool:
        ts = datetime.now().strftime("%H:%M:%S")
        text = (
            f"📈 <b>Oportunidad GGAL</b> [{ts}]\n"
            f"Estrategia: <b>{result.strategy}</b>\n"
            f"Strike: {result.strike:,.0f}  |  Días: {result.days}\n"
            f"TNA: <b>{result.tna_pct:.1f}%</b>  |  "
            f"TEA: {result.tea_pct:.1f}%\n"
            f"Spread vs caución: <b>+{result.spread_vs_caucion_pct:.1f}%</b>\n"
            f"Colchón: {result.cushion_pct:.1f}%  |  {result.moneyness}\n"
            f"Prima: {result.premium:,.2f} ARS"
        )
        return self._send(text)

    def send_top_opportunities(self, results: list[RateResult], n: int = 3) -> bool:
        if not results:
            return False
        lines = [f"📊 <b>Top {min(n, len(results))} — GGAL</b>"]
        for r in results[:n]:
            lines.append(
                f"• {r.strategy} K={r.strike:,.0f}  "
                f"TNA={r.tna_pct:.1f}%  (+{r.spread_vs_caucion_pct:.1f}%)  "
                f"col={r.cushion_pct:.1f}%  {r.moneyness}"
            )
        return self._send("\n".join(lines))

    def send_swing_target(
        self,
        symbol: str,
        capture_pct: float,
        target_pct: float,
        strategy: str,
        days_held: int,
    ) -> bool:
        """Alerta de objetivo de swing alcanzado."""
        strat_label = "Lanzamiento cubierto" if strategy == "COVERED_CALL" else "Venta de put"
        ts = datetime.now().strftime("%H:%M:%S")
        text = (
            f"<b>Objetivo de salida alcanzado [{ts}]</b>\n"
            f"Posición: <b>{symbol}</b> — {strat_label}\n"
            f"Captura actual: <b>{capture_pct:.1f}%</b> "
            f"(objetivo: {target_pct:.0f}%)\n"
            f"Días en posición: {days_held}\n"
            f"<i>Revisá la posición y ejecutá el cierre en Bull Market.</i>"
        )
        return self._send(text)

    def send_recommendation(self, rec: "Recommendation") -> bool:  # noqa: F821
        """Envía la recomendacion de un perfil por Telegram."""
        from optionsdesk.signals.recommender import Recommendation  # local para evitar ciclo
        strat = (
            "Lanzamiento cubierto"
            if rec.result.strategy == "COVERED_CALL"
            else "Venta de put"
        )
        light_labels = {
            "verde": "APROBADA",
            "amarillo": "CON ADVERTENCIAS",
            "rojo": "RECHAZADA",
        }
        ts = datetime.now().strftime("%H:%M:%S")
        text = (
            f"<b>Recomendacion {rec.profile} [{ts}]</b>\n"
            f"Estado: <b>{light_labels.get(rec.light, rec.light)}</b>\n"
            f"Estrategia: {strat} — <b>{rec.result.symbol}</b>\n"
            f"TNA: <b>{rec.result.tna_pct:.1f}%</b>  |  "
            f"Colchon: {rec.result.cushion_pct:.1f}%\n"
            f"Probabilidad: {rec.success_probability * 100:.0f}%  |  "
            f"Score: {rec.score:.0f}/100\n"
            f"\n{rec.plain_explanation}"
        )
        return self._send(text)
