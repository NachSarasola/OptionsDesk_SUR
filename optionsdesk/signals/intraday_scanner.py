"""Detector de oportunidades intradía en la cadena de opciones GGAL.

Lee los snapshots del recorder (15s por default) y separa los movimientos
de prima impulsados por IV (edge real para el vendedor) de los explicados por
el delta (movimiento del subyacente).

Concepto:
    Δpremium_total   = Δpremium_delta  + Δpremium_iv
    Δpremium_delta   ≈ delta × Δspot
    Δpremium_iv      = residual

Un z-score alto del residual IV → la prima subió "más de lo que el spot explica"
→ buena ventana para lanzar (WRITE). Para posiciones cortas abiertas, una caída
brusca del residual → la prima se desplomó → cerrar rápido (CLOSE).

Limitaciones honestas (visibles en la UI):
    - Primary permite streaming; IOL REST usa frecuencia acotada; BYMA Open Data queda demorado.
    - Liquidez fina en opciones GGAL → un único print puede generar falsas señales.
    - Solo funciona con el recorder corriendo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from optionsdesk.signals.monitor import OpenPosition

logger = logging.getLogger(__name__)

# Ventana de lookback (número de snapshots) para calcular el cambio de prima
_WINDOW_SNAPSHOTS = 3     # ~6 min con interval=120s
# Umbral de z-score para considerar un spike de IV accionable
_ZSCORE_WRITE_THRESHOLD = 2.0
# Caída de prima (%) para señal de cierre en posición corta
_CLOSE_DROP_PCT = 30.0
# Mínimo de snapshots para que haya serie útil
_MIN_SNAPSHOTS = 5


@dataclass
class IntradaySignal:
    signal_type: str     # "WRITE" | "CLOSE"
    symbol: str
    option_type: str     # "C" | "P"
    current_premium: float
    premium_change_pct: float
    iv_residual_z: float
    message: str


# ── Carga de snapshots ────────────────────────────────────────────────────────

def _load_day_snapshots(snapshots_dir: Path, day: date) -> pd.DataFrame:
    """Carga todos los parquets del día en un único DataFrame.

    Columnas esperadas: timestamp, symbol, spot, bid, ask, last, volume, delta.
    La columna delta puede no existir (recorder viejo) → se rellena con NaN.
    """
    day_dir = snapshots_dir / day.isoformat()
    if not day_dir.exists():
        logger.debug("No hay snapshots para %s en %s", day, snapshots_dir)
        return pd.DataFrame()

    files = sorted(day_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            logger.warning("Error leyendo snapshot %s: %s", f.name, exc)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "delta" not in df.columns:
        df["delta"] = float("nan")
    return df


def load_intraday_chain(
    snapshots_dir: str | Path,
    day: date,
) -> dict[str, pd.Series]:
    """Devuelve {symbol: Serie(mid) indexada por timestamp}.

    Solo incluye símbolos con al menos _MIN_SNAPSHOTS puntos.
    """
    df = _load_day_snapshots(Path(snapshots_dir), day)
    if df.empty:
        return {}

    def mid(row: pd.Series) -> float:
        b, a = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
        if b > 0 and a > 0:
            return (b + a) / 2.0
        return float(row.get("last", 0) or b or a)

    result: dict[str, pd.Series] = {}
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp")
        mids = grp.apply(mid, axis=1)
        mids.index = pd.to_datetime(grp["timestamp"].values)
        mids = mids[mids > 0]
        if len(mids) >= _MIN_SNAPSHOTS:
            result[str(symbol)] = mids

    return result


# ── Descomposición delta / IV ─────────────────────────────────────────────────

def _compute_residuals(
    premium_series: pd.Series,
    spot_series: pd.Series,
    delta_series: pd.Series,
    window: int = _WINDOW_SNAPSHOTS,
) -> pd.Series:
    """Residual de IV (Δpremium - delta × Δspot) en cada snapshot."""
    dp = premium_series.diff(window)
    ds = spot_series.diff(window)

    # Reindexar delta al mismo índice que los precios
    delta_aligned = delta_series.reindex(premium_series.index, method="nearest")

    delta_explained = delta_aligned * ds
    residual = dp - delta_explained
    return residual.dropna()


def _zscore_series(series: pd.Series) -> pd.Series:
    """Z-score robusto. Usa MAD cuando la dispersión es suficiente; cae a std si no.

    El MAD falla cuando la mayoría de observaciones son idénticas (MAD → 0)
    mientras hay outliers claros — exactamente el patrón de un spike de IV
    sobre una base estable. En ese caso std captura el outlier correctamente.
    """
    med = series.median()
    mad = (series - med).abs().median()
    if mad >= 1e-6:
        return (series - med) / (mad * 1.4826)

    # Fallback: z-score estándar
    mu  = series.mean()
    std = series.std(ddof=1)
    if std < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / std


# ── API pública ───────────────────────────────────────────────────────────────

def detect_write_opportunities(
    snapshots_dir: str | Path,
    day: date,
    zscore_threshold: float = _ZSCORE_WRITE_THRESHOLD,
    window: int = _WINDOW_SNAPSHOTS,
) -> list[IntradaySignal]:
    """Detecta primas con spike de IV (buena ventana para lanzar).

    Solo genera señal en el snapshot más reciente de cada símbolo.
    """
    df = _load_day_snapshots(Path(snapshots_dir), day)
    if df.empty:
        return []

    # Serie de spot intradía (una por timestamp)
    spot_by_ts: dict[pd.Timestamp, float] = {}
    for ts, grp in df.groupby("timestamp"):
        vals = grp["spot"].dropna()
        if not vals.empty:
            spot_by_ts[pd.Timestamp(ts)] = float(vals.iloc[0])

    if len(spot_by_ts) < _MIN_SNAPSHOTS:
        return []

    spot_series = pd.Series(spot_by_ts).sort_index()
    signals: list[IntradaySignal] = []

    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp")
        grp.index = pd.to_datetime(grp["timestamp"].values)

        def mid(row: pd.Series) -> float:
            b, a = float(row.get("bid", 0) or 0), float(row.get("ask", 0) or 0)
            if b > 0 and a > 0:
                return (b + a) / 2.0
            return float(row.get("last", 0) or b or a)

        mids = grp.apply(mid, axis=1)
        mids = mids[mids > 0]
        if len(mids) < _MIN_SNAPSHOTS:
            continue

        # Delta serie — puede estar ausente o ser NaN
        if "delta" not in grp.columns:
            continue
        delta_series = pd.to_numeric(grp["delta"], errors="coerce").reindex(mids.index)
        if delta_series.isna().any():
            continue

        # Spot alineado al mismo índice
        spot_aligned = spot_series.reindex(mids.index, method="nearest").ffill()

        residuals = _compute_residuals(mids, spot_aligned, delta_series, window)
        if len(residuals) < 3:
            continue

        zscores = _zscore_series(residuals)
        last_z = float(zscores.iloc[-1]) if len(zscores) else 0.0
        last_mid = float(mids.iloc[-1])

        # Cambio % total de la prima en la ventana
        if len(mids) >= window + 1:
            prev_mid = float(mids.iloc[-(window + 1)])
            change_pct = (last_mid - prev_mid) / prev_mid * 100.0 if prev_mid > 0 else 0.0
        else:
            change_pct = 0.0

        if last_z >= zscore_threshold and change_pct > 0:
            opt_type = "C" if "C" in str(symbol).upper() else "P"
            elapsed_min = (
                (mids.index[-1] - mids.index[-(window + 1)]).total_seconds() / 60.0
                if len(mids) >= window + 1
                else 0.0
            )
            signals.append(IntradaySignal(
                signal_type="WRITE",
                symbol=str(symbol),
                option_type=opt_type,
                current_premium=last_mid,
                premium_change_pct=change_pct,
                iv_residual_z=last_z,
                message=(
                    f"Prima subio {change_pct:+.1f}% en {elapsed_min:.0f} min - componente IV z={last_z:.1f} "
                    f"(vs umbral {zscore_threshold:.1f}). "
                    "La prima luce rica; ventana para lanzar. "
                    "Verificá liquidez antes de operar."
                ),
            ))

    return sorted(signals, key=lambda s: s.iv_residual_z, reverse=True)


def detect_close_opportunities(
    snapshots_dir: str | Path,
    day: date,
    open_positions: list["OpenPosition"],
    drop_threshold_pct: float = _CLOSE_DROP_PCT,
    window: int = _WINDOW_SNAPSHOTS,
) -> list[IntradaySignal]:
    """Detecta posiciones cortas cuya prima se desplomó intradía.

    Una caída de prima ≥ drop_threshold_pct en la ventana → capturaste
    X% en pocas horas → señal de cierre anticipado.
    """
    if not open_positions:
        return []

    chain = load_intraday_chain(snapshots_dir, day)
    if not chain:
        return []

    pos_by_symbol = {p.symbol: p for p in open_positions}
    signals: list[IntradaySignal] = []

    for symbol, series in chain.items():
        pos = pos_by_symbol.get(symbol)
        if pos is None:
            continue

        if len(series) < window + 1:
            continue

        last_mid = float(series.iloc[-1])
        prev_mid = float(series.iloc[-(window + 1)])

        if prev_mid <= 0:
            continue

        change_pct = (last_mid - prev_mid) / prev_mid * 100.0

        # Caída significativa en la ventana → captura rápida
        if change_pct <= -drop_threshold_pct:
            captured = (
                (pos.premium_received - last_mid) / pos.premium_received * 100.0
                if pos.premium_received > 0 else 0.0
            )
            opt_type = pos.opt_type
            elapsed_min = (
                (series.index[-1] - series.index[-(window + 1)]).total_seconds() / 60.0
            )
            signals.append(IntradaySignal(
                signal_type="CLOSE",
                symbol=symbol,
                option_type=opt_type,
                current_premium=last_mid,
                premium_change_pct=change_pct,
                iv_residual_z=0.0,   # no descomponemos para cierre — la caída sola alcanza
                message=(
                    f"Prima cayo {change_pct:.1f}% en los ultimos {elapsed_min:.0f} min. "
                    f"Captura estimada: {captured:.0f}% de la prima original. "
                    "Considerar cierre anticipado para asegurar la ganancia."
                ),
            ))

    return sorted(signals, key=lambda s: s.premium_change_pct)
