"""Preflight de operacion manual: valida datos y configuracion sin enviar ordenes."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from optionsdesk.config.costs import DEFAULT_COSTS
from optionsdesk.config.settings import settings
from optionsdesk.data import recorder

_REALTIME_SOURCES = {"PRIMARY", "IOL"}


def _line(level: str, label: str, detail: str) -> None:
    print(f"[{level:<4}] {label}: {detail}")


def main() -> int:
    now = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    market_open = recorder._is_byma_market_session(now)
    failures = 0

    print(f"OptionsDesk preflight read-only | {now.isoformat(timespec='seconds')}")
    _line("INFO", "rueda BYMA", "abierta" if market_open else "fuera de 11:00-17:00")

    if settings.costs_verified:
        _line("OK", "tarifario", settings.costs_profile)
    else:
        failures += 1
        _line("FAIL", "tarifario", "falta COSTS_VERIFIED=true con tasas confirmadas del ALyC")
    _line(
        "INFO",
        "costos usados",
        (
            f"comision acciones={DEFAULT_COSTS.stock_commission_pct:.3f}% "
            f"opciones={DEFAULT_COSTS.option_commission_pct:.3f}% "
            f"ejercicio={DEFAULT_COSTS.exercise_fee_pct:.3f}% | "
            f"derechos acciones={DEFAULT_COSTS.stock_market_fee_pct:.3f}% "
            f"opciones={DEFAULT_COSTS.option_market_fee_pct:.3f}% "
            f"ejercicio={DEFAULT_COSTS.exercise_market_fee_pct:.3f}% | "
            f"IVA={DEFAULT_COSTS.iva_rate * 100:.1f}%"
        ),
    )

    if settings.scalping_manual_sizing:
        detail = (
            f"manual activo; capital referencia ${settings.default_capital:,.0f}"
            if settings.default_capital and settings.default_capital > 0
            else "manual activo; capital no requerido para emitir ticket"
        )
        _line("OK", "sizing", detail)
    elif settings.default_capital and settings.default_capital > 0:
        _line("OK", "capital sizing", f"${settings.default_capital:,.0f}")
    else:
        _line("WARN", "capital sizing", "cargalo en el panel antes de aceptar una senal")

    provider = None
    try:
        provider = recorder._build_provider()
        chain = provider.get_options_chain()
        health = provider.get_health()
        realtime = health.source in _REALTIME_SOURCES and health.connected
        if realtime:
            _line("OK", "feed", f"{health.source} conectado")
        else:
            failures += 1
            _line("FAIL", "feed", f"{health.source} no es realtime o no esta conectado")

        if chain is None:
            failures += 1
            _line("FAIL", "cadena", health.last_error or "sin datos")
        else:
            tradeable = [
                q for q in chain.options.values()
                if q.bid > 0 and q.ask > 0 and not q.is_stale(settings.scalping_max_quote_age_s)
            ]
            _line(
                "OK" if tradeable else ("WARN" if not market_open else "FAIL"),
                "cadena",
                f"GGAL={chain.spot.mid:,.2f} opciones={len(chain.options)} puntas_frescas={len(tradeable)}",
            )
            if market_open and not tradeable:
                failures += 1
    except Exception as exc:
        failures += 1
        _line("FAIL", "feed", f"{type(exc).__name__}: {exc}")
    finally:
        if provider is not None:
            provider.disconnect()

    if settings.is_primary_configured():
        _line("OK", "Primary", "configurado")
    elif settings.is_iol_configured():
        _line("WARN", "Primary", "pendiente; IOL REST queda como fallback de menor frecuencia")
    else:
        _line("WARN", "Primary", "pendiente; configura Primary o IOL para datos vivos")
    if settings.telegram_token and settings.telegram_chat_id:
        _line("OK", "Telegram", "configurado")
    else:
        _line("WARN", "Telegram", "opcional, no configurado")
    _line(
        "INFO",
        "recorder",
        f"cadena IOL cada {settings.iol_recorder_interval_s}s; spot tape cada {settings.spot_tape_interval_s}s",
    )

    from optionsdesk.signals.monitor import PositionMonitor

    open_positions = PositionMonitor(settings.open_positions_file).load_positions()
    if open_positions:
        failures += 1
        _line(
            "FAIL",
            "posiciones locales",
            f"{len(open_positions)} abierta(s); cerrar/archivar con fill real antes de otro scalp",
        )
    else:
        _line("OK", "posiciones locales", "0")

    print("Resultado: " + ("LISTO" if failures == 0 else f"BLOQUEADO ({failures} requisito(s))"))
    print("Este comando nunca envia ordenes. La ejecucion real se confirma manualmente en Matriz.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
