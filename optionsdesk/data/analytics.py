"""Analytics histórico sobre snapshots acumulados por el recorder.

Lee los parquets de data/snapshots/**/*.parquet vía DuckDB sin migración
ni base de datos en disco. Cada función retorna un DataFrame listo para
st.dataframe / st.line_chart / st.bar_chart.

Si no hay snapshots suficientes → devuelve DataFrame vacío (no lanza excepción).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def connect(snapshots_dir: Path) -> "duckdb.DuckDBPyConnection":  # type: ignore[name-defined]
    """Crea una conexión DuckDB con una VIEW lazy sobre todos los parquets.

    La VIEW se llama 'snaps' y tiene columnas:
        timestamp, symbol, spot, bid, ask, last, volume
    """
    import duckdb  # import local: no rompe si duckdb no está instalado en tests unitarios

    con = duckdb.connect()
    parquet_glob = str(snapshots_dir / "**" / "*.parquet")
    try:
        # hive_partitioning=false: los dirs son fechas, no particiones Hive
        con.execute(
            f"CREATE VIEW snaps AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=false)"
        )
        # Verificar que hay al menos 1 fila (falla silenciosamente si no hay archivos)
        con.execute("SELECT count(*) FROM snaps").fetchone()
    except Exception as e:
        logger.debug("DuckDB VIEW snaps fallo (sin snapshots?): %s", e)
        # Vista vacía con el mismo schema para que las funciones no exploten
        con.execute(
            "CREATE VIEW snaps AS "
            "SELECT * FROM (VALUES "
            "(TIMESTAMPTZ '1970-01-01', '', 0.0, 0.0, 0.0, 0.0, 0.0)) "
            "t(timestamp, symbol, spot, bid, ask, last, volume) "
            "WHERE 1=0"
        )
    return con


def _cutoff(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def premium_evolution(
    con: "duckdb.DuckDBPyConnection",
    symbol: str,
    days: int = 7,
) -> pd.DataFrame:
    """Serie temporal del mid price de una opción específica.

    Retorna columnas: ts (datetime), mid (float).
    Agrupa por hora para reducir ruido de snapshots a 15s.
    """
    try:
        cutoff = _cutoff(days)
        query = """
        SELECT
            date_trunc('hour', timestamp::TIMESTAMPTZ) AS ts,
            avg((bid + ask) / 2.0) AS mid
        FROM snaps
        WHERE symbol = ?
          AND timestamp::TIMESTAMPTZ >= ?
          AND bid > 0 AND ask > 0
        GROUP BY 1
        ORDER BY 1
        """
        df = con.execute(query, [symbol, cutoff]).df()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")
    except Exception as e:
        logger.debug("analytics query fallo: %s", e)
        return pd.DataFrame()


def spot_history(
    con: "duckdb.DuckDBPyConnection",
    days: int = 30,
) -> pd.DataFrame:
    """Evolución del spot de GGAL registrado en los snapshots.

    Retorna index=ts (datetime), columna spot.
    """
    try:
        cutoff = _cutoff(days)
        query = """
        SELECT
            date_trunc('hour', timestamp::TIMESTAMPTZ) AS ts,
            avg(spot) AS spot
        FROM snaps
        WHERE timestamp::TIMESTAMPTZ >= ?
          AND spot > 0
        GROUP BY 1
        ORDER BY 1
        """
        df = con.execute(query, [cutoff]).df()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")
    except Exception as e:
        logger.debug("analytics query fallo: %s", e)
        return pd.DataFrame()


def tna_proxy_heatmap(
    con: "duckdb.DuckDBPyConnection",
    days: int = 14,
    top_n: int = 20,
) -> pd.DataFrame:
    """Mapa de calor de prima normalizada (mid/spot en %) por símbolo y fecha.

    No es TNA exacta (requiere días al vencimiento que no están en el snapshot)
    sino una proxy del nivel relativo de la prima. Útil para detectar spikes
    de IV comparando la misma opción a lo largo del tiempo.

    Retorna un DataFrame pivotado: filas=símbolo, columnas=fecha.
    """
    try:
        cutoff = _cutoff(days)
        query = """
        SELECT
            symbol,
            date_trunc('day', timestamp::TIMESTAMPTZ)::DATE AS date,
            avg(((bid + ask) / 2.0) / spot * 100.0) AS mid_pct
        FROM snaps
        WHERE timestamp::TIMESTAMPTZ >= ?
          AND bid > 0 AND ask > 0 AND spot > 0
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
        df = con.execute(query, [cutoff]).df()
        if df.empty:
            return df

        # Quedarse con los top_n símbolos de mayor prima promedio (más activos)
        top_syms = (
            df.groupby("symbol")["mid_pct"]
            .mean()
            .nlargest(top_n)
            .index.tolist()
        )
        df = df[df["symbol"].isin(top_syms)]

        pivot = df.pivot(index="symbol", columns="date", values="mid_pct").round(2)
        pivot.columns = [str(c) for c in pivot.columns]
        return pivot
    except Exception as e:
        logger.debug("analytics query fallo: %s", e)
        return pd.DataFrame()


def volume_by_symbol(
    con: "duckdb.DuckDBPyConnection",
    days: int = 7,
    top_n: int = 30,
) -> pd.DataFrame:
    """Top opciones por volumen acumulado en los últimos N días.

    Retorna columnas: symbol, total_volume, avg_mid, snapshots.
    """
    try:
        cutoff = _cutoff(days)
        query = """
        SELECT
            symbol,
            sum(volume)               AS total_volume,
            avg((bid + ask) / 2.0)    AS avg_mid,
            count(*)                  AS snapshots
        FROM snaps
        WHERE timestamp::TIMESTAMPTZ >= ?
          AND volume > 0
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT ?
        """
        return con.execute(query, [cutoff, top_n]).df()
    except Exception as e:
        logger.debug("analytics query fallo: %s", e)
        return pd.DataFrame()


def available_symbols(
    con: "duckdb.DuckDBPyConnection",
    days: int = 7,
    min_snapshots: int = 3,
) -> list[str]:
    """Lista de símbolos con al menos min_snapshots registros en los últimos N días."""
    try:
        cutoff = _cutoff(days)
        query = """
        SELECT symbol
        FROM snaps
        WHERE timestamp::TIMESTAMPTZ >= ?
          AND bid > 0 AND ask > 0
        GROUP BY symbol
        HAVING count(*) >= ?
        ORDER BY sum(volume) DESC NULLS LAST
        LIMIT 50
        """
        rows = con.execute(query, [cutoff, min_snapshots]).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception as e:
        logger.debug("analytics query fallo: %s", e)
        return []


def snapshot_count(con: "duckdb.DuckDBPyConnection") -> int:
    """Total de registros en la vista. 0 si está vacía o falla."""
    try:
        return con.execute("SELECT count(*) FROM snaps").fetchone()[0]  # type: ignore[index]
    except Exception as e:
        logger.debug("snapshot_count fallo: %s", e)
        return 0
