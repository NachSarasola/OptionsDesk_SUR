"""Smoke test read-only para habilitaciones Primary / Matriz OMS."""
from __future__ import annotations

import argparse
import sys

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.primary import PrimaryProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida auth, discovery y market data WebSocket sin enviar ordenes."
    )
    parser.add_argument("--wait-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if not settings.is_primary_configured():
        print("Faltan PRIMARY_BASE_URL, PRIMARY_USER y PRIMARY_PASSWORD.", file=sys.stderr)
        return 2

    provider = PrimaryProvider()
    try:
        provider.connect()
        provider.wait_until_ready(timeout_s=args.wait_seconds)
        chain = provider.get_options_chain()
        health = provider.get_health()
        if chain is None:
            print(f"Sin cadena realtime. Ultimo error: {health.last_error or '-'}", file=sys.stderr)
            return 1
        print("Primary read-only OK")
        print(f"Feed: {health.source} | conectado: {health.connected}")
        print(f"GGAL: {chain.spot.mid:,.2f}")
        print(
            f"Opciones recibidas: {len(chain.options)} | "
            f"operables bid/ask: {health.options_tradeable}"
        )
        print(f"Caucion TNA: {provider.get_caucion_tna():.2f}%")
        return 0
    finally:
        provider.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
