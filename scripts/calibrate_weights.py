#!/usr/bin/env python3
"""GridSearch de _WEIGHTS del perfil EQUILIBRADO optimizando Sharpe anualizado.

Usa los snapshots acumulados por el recorder para simular la estrategia con
distintas combinaciones de pesos y reporta las top 10 configuraciones por Sharpe.

Uso:
    python scripts/calibrate_weights.py --days 30 --capital 500000

Notas:
    - Cada combinación de pesos corre simulate_strategy() completo — puede tardar
      varios minutos dependiendo de la cantidad de snapshots.
    - El resultado se guarda en data/calibration_results.parquet.
    - NO modifica _WEIGHTS automáticamente. El usuario decide si actualizar.
"""
from __future__ import annotations

import argparse
import itertools
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING)

_KEYS = ["ev", "vol", "cushion", "probability", "liquidity"]
_GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
_SUM_TARGET = 1.0
_TOLERANCE = 0.001


def _generate_weight_combos() -> list[dict[str, float]]:
    """Genera todas las combinaciones de pesos que suman 1.0."""
    combos: list[dict[str, float]] = []
    for vals in itertools.product(_GRID, repeat=len(_KEYS)):
        if abs(sum(vals) - _SUM_TARGET) < _TOLERANCE:
            combos.append(dict(zip(_KEYS, vals)))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="GridSearch de pesos del recommender")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Días hacia atrás desde hoy para la simulación (default: 30)",
    )
    parser.add_argument(
        "--capital", type=float, default=500_000.0,
        help="Capital inicial en ARS (default: 500000)",
    )
    parser.add_argument(
        "--caucion-tna", type=float, default=60.0,
        help="TNA de caución para el Sharpe de referencia (default: 60.0)",
    )
    parser.add_argument(
        "--snapshots-dir", type=str, default="data/snapshots",
        help="Directorio de snapshots (default: data/snapshots)",
    )
    parser.add_argument(
        "--mode", type=str, default="bid_ask", choices=["bid_ask", "mid"],
        help="Modo de ejecución: bid_ask (realista) o mid (optimista)",
    )
    args = parser.parse_args()

    end   = date.today()
    start = end - timedelta(days=args.days)
    snap_dir = Path(args.snapshots_dir)

    if not snap_dir.exists() or not any(snap_dir.iterdir()):
        print(f"ERROR: No hay snapshots en {snap_dir}. Ejecutá el recorder primero.")
        return

    from optionsdesk.backtest.portfolio_sim import simulate_strategy
    from optionsdesk.backtest.metrics import compute_metrics, sharpe_ratio
    from optionsdesk.signals.recommender import RiskProfile

    combos = _generate_weight_combos()
    print(f"Calibrando {len(combos)} combinaciones de pesos sobre {args.days} días "
          f"({start} → {end}), modo={args.mode} ...")

    results: list[dict] = []
    for i, weights in enumerate(combos, 1):
        if i % 20 == 0:
            print(f"  [{i}/{len(combos)}] ...")
        try:
            sim = simulate_strategy(
                snapshots_dir=snap_dir,
                profile=RiskProfile.EQUILIBRADO,
                start=start.isoformat(),
                end=end.isoformat(),
                capital_initial=args.capital,
                execution_mode=args.mode,
                caucion_tna_pct=args.caucion_tna,
                weights_override=weights,
            )

            if len(sim.pnl_series) < 2:
                continue   # sin trades suficientes para métricas significativas

            m = compute_metrics(
                sim.pnl_series,
                sim.equity_curve,
                caucion_tna_pct=args.caucion_tna,
            )

            results.append({
                **weights,
                "sharpe":          m["sharpe"],
                "sortino":         m["sortino"],
                "profit_factor":   m["profit_factor"],
                "win_rate_pct":    m["win_rate_pct"],
                "cagr_pct":        m["cagr_pct"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "n_trades":        m["n_trades"],
                "total_pnl_ars":   m["total_pnl_ars"],
            })
        except Exception as exc:
            logging.debug("Combo %s falló: %s", weights, exc)

    if not results:
        print("Sin resultados — verificá que hay snapshots suficientes en el rango.")
        return

    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    out_path = Path("data/calibration_results.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nResultados guardados en {out_path}")
    print(f"\nTop 10 por Sharpe ({args.mode} mode):")
    print(df.head(10).to_string(index=False))

    # Mostrar los pesos actuales del EQUILIBRADO vs el top
    from optionsdesk.signals.recommender import _WEIGHTS, RiskProfile as RP
    current = _WEIGHTS[RP.EQUILIBRADO]
    print(f"\nPesos actuales (EQUILIBRADO): {current}")
    best = df.iloc[0]
    print(f"Pesos óptimos encontrados:    "
          f"{{ev:{best.ev:.2f}, vol:{best.vol:.2f}, cushion:{best.cushion:.2f}, "
          f"probability:{best.probability:.2f}, liquidity:{best.liquidity:.2f}}}")
    print(f"  → Sharpe={best.sharpe:.2f}  WinRate={best.win_rate_pct:.0f}%  "
          f"PF={best.profit_factor:.2f}  CAGR={best.cagr_pct:.1f}%")
    print("\nPara actualizar los pesos editá _WEIGHTS en signals/recommender.py")


if __name__ == "__main__":
    main()
