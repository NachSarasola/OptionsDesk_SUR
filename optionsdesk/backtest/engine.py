"""Motor de backtest: reproduce snapshots del recorder para evaluar estrategias.

Al principio, el "backtest" es en realidad forward-test/paper: los snapshots
acumulados por el recorder son el dataset. A las pocas semanas de correr el
recorder en horario de mercado hay suficiente historia para análisis.

Uso:
    engine = BacktestEngine()
    df = engine.run("2026-05-20", "2026-05-21", caucion_tna_pct=115.0)
    print(df.sort_values("spread_vs_caucion_pct", ascending=False).head(20))
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from optionsdesk.core.benchmark import Benchmark
from optionsdesk.core.instruments import load_expiry_calendar
from optionsdesk.data.providers.base import OptionsChain, Quote
from optionsdesk.strategies.covered_call import CoveredCallConfig, CoveredCallScanner
from optionsdesk.strategies.short_put import ShortPutConfig, ShortPutScanner

logger = logging.getLogger(__name__)


def _reconstruct_chain(
    rows: pd.DataFrame, ts: pd.Timestamp
) -> Optional[OptionsChain]:
    """Reconstruye un OptionsChain desde las filas de un snapshot."""
    snap = rows[rows["timestamp"] == ts]
    if snap.empty:
        return None

    spot_val = float(snap["spot"].iloc[0])
    spot_q = Quote(
        symbol="GGAL",
        bid=spot_val * 0.999, ask=spot_val * 1.001, last=spot_val,
        volume=0, timestamp=ts.to_pydatetime(),
    )
    opts: dict[str, Quote] = {
        row["symbol"]: Quote(
            symbol=row["symbol"],
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            last=float(row["last"]),
            volume=float(row["volume"]),
            timestamp=ts.to_pydatetime(),
        )
        for _, row in snap.iterrows()
    }
    return OptionsChain(underlying="GGAL", spot=spot_q, options=opts)


class BacktestEngine:
    """Reproduce snapshots del recorder para simular el desempeño de estrategias."""

    def __init__(
        self,
        snapshots_dir: str | Path = "data/snapshots",
        cc_config: Optional[CoveredCallConfig] = None,
        sp_config: Optional[ShortPutConfig] = None,
        expiry_calendar: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self._dir = Path(snapshots_dir)
        self._cc_cfg = cc_config or CoveredCallConfig()
        self._sp_cfg = sp_config or ShortPutConfig()
        self._expiry_cal = expiry_calendar or {}
        if not self._expiry_cal:
            try:
                self._expiry_cal = load_expiry_calendar(config_path)
            except Exception:
                logger.warning("No se pudo cargar el calendario de vencimientos.")

    def run(
        self,
        date_from: str,
        date_to: str,
        caucion_tna_pct: float = 115.0,
    ) -> pd.DataFrame:
        """Corre el backtest sobre el rango de fechas dado.

        Devuelve un DataFrame con todas las oportunidades detectadas en cada
        snapshot, ordenadas por timestamp y spread vs caución.
        """
        benchmark = Benchmark(caucion_tna_pct=caucion_tna_pct, days=30)
        cc_scanner = CoveredCallScanner(self._cc_cfg, self._expiry_cal)
        sp_scanner = ShortPutScanner(self._sp_cfg, self._expiry_cal)

        all_rows: list[dict] = []

        for date_str in pd.date_range(date_from, date_to, freq="D").strftime("%Y-%m-%d"):
            day_dir = self._dir / date_str
            if not day_dir.exists():
                continue
            files = sorted(day_dir.glob("*.parquet"))
            if not files:
                continue

            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

            for ts in df["timestamp"].unique():
                chain = _reconstruct_chain(df, ts)
                if chain is None:
                    continue

                for result in cc_scanner.scan(chain, benchmark):
                    result.set_spread(caucion_tna_pct)
                    all_rows.append({
                        "date": date_str, "ts": ts,
                        "symbol": result.symbol, "strategy": result.strategy,
                        "strike": result.strike, "days": result.days,
                        "tna_pct": result.tna_pct, "tea_pct": result.tea_pct,
                        "spread_pct": result.spread_vs_caucion_pct,
                        "cushion_pct": result.cushion_pct,
                        "moneyness": result.moneyness,
                        "delta": result.delta, "iv": result.iv,
                    })

                for result in sp_scanner.scan(chain, benchmark):
                    result.set_spread(caucion_tna_pct)
                    all_rows.append({
                        "date": date_str, "ts": ts,
                        "symbol": result.symbol, "strategy": result.strategy,
                        "strike": result.strike, "days": result.days,
                        "tna_pct": result.tna_pct, "tea_pct": result.tea_pct,
                        "spread_pct": result.spread_vs_caucion_pct,
                        "cushion_pct": result.cushion_pct,
                        "moneyness": result.moneyness,
                        "delta": result.delta, "iv": result.iv,
                    })

        if not all_rows:
            logger.warning("No hay snapshots para el rango %s - %s.", date_from, date_to)
            return pd.DataFrame()

        return (
            pd.DataFrame(all_rows)
            .sort_values(["ts", "spread_pct"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def run_with_exit(
        self,
        entry_date: str,
        exit_date: str,
        caucion_tna_pct: float = 115.0,
        t_star_days: int = 15,
    ) -> pd.DataFrame:
        """Compara 'salir en t* días' vs 'hold a vencimiento' usando snapshots reales.

        Requiere datos del recorder para las dos fechas. Si no hay snapshots,
        devuelve un DataFrame vacío con un warning.

        Columnas del resultado:
          symbol, strategy, entry_ts, exit_ts, premium_entry, premium_exit,
          pnl_swing, pnl_hold_est, swing_is_better
        """
        from optionsdesk.core.pricing import crr_price
        import math

        df_entry = self._load_range(entry_date, entry_date)
        df_exit  = self._load_range(exit_date,  exit_date)

        if df_entry is None or df_exit is None or df_entry.empty or df_exit.empty:
            logger.warning(
                "run_with_exit: sin snapshots para entry=%s o exit=%s",
                entry_date, exit_date,
            )
            return pd.DataFrame()

        # Toma el primer snapshot de cada día
        entry_ts = df_entry["ts"].min()
        exit_ts  = df_exit["ts"].min()

        snap_entry = df_entry[df_entry["ts"] == entry_ts]
        snap_exit  = df_exit[df_exit["ts"] == exit_ts]

        rows = []
        for _, row_e in snap_entry.iterrows():
            sym = row_e.get("symbol")
            if sym is None or not isinstance(sym, str):
                continue
            match_e = snap_exit[snap_exit["symbol"] == sym]
            if match_e.empty:
                continue
            row_x = match_e.iloc[0]

            # Prima de entrada = mid de la opción en snapshot de entrada
            bid_e = float(row_e.get("bid", 0))
            ask_e = float(row_e.get("ask", 0))
            prem_entry = (bid_e + ask_e) / 2.0 if bid_e > 0 and ask_e > 0 else bid_e or ask_e

            bid_x = float(row_x.get("bid", 0))
            ask_x = float(row_x.get("ask", 0))
            prem_exit = (bid_x + ask_x) / 2.0 if bid_x > 0 and ask_x > 0 else bid_x or ask_x

            if prem_entry <= 0:
                continue

            pnl_swing   = prem_entry - prem_exit        # cerrar al precio de mercado
            pnl_hold_est = prem_entry                   # estimado si expira OTM

            rows.append({
                "symbol":        sym,
                "strategy":      row_e.get("strategy", ""),
                "entry_ts":      entry_ts,
                "exit_ts":       exit_ts,
                "premium_entry": round(prem_entry, 2),
                "premium_exit":  round(prem_exit, 2),
                "pnl_swing":     round(pnl_swing, 2),
                "pnl_hold_est":  round(pnl_hold_est, 2),
                "swing_is_better": pnl_swing >= pnl_hold_est * 0.80,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values("pnl_swing", ascending=False).reset_index(drop=True)

    def _load_range(self, date_from: str, date_to: str) -> Optional[pd.DataFrame]:
        """Carga snapshots del rango en un único DataFrame."""
        frames = []
        for date_str in pd.date_range(date_from, date_to, freq="D").strftime("%Y-%m-%d"):
            day_dir = self._dir / date_str
            if not day_dir.exists():
                continue
            files = sorted(day_dir.glob("*.parquet"))
            if files:
                frames.append(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
        return pd.concat(frames, ignore_index=True) if frames else None
