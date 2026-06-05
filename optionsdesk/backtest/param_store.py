"""Registro y persistencia de los parametros tuneables del sistema de señales.

El learner (param_learner.py) busca mejores valores para estos parametros a medida
que la demo opera; este modulo es la fuente de verdad de:

  1. QUE parametros se pueden tunear, con sus limites y escala de perturbacion.
  2. A QUE setups afecta cada parametro (para asignar credito cuando un setup falla).
  3. El set de parametros ACTIVO (campeon o retador desplegado), persistido en
     data/learned_params.json, que las funciones de señal leen en runtime.

Diseño clave: si no existe el archivo de parametros aprendidos, `param()` devuelve
el default pasado por el caller — el comportamiento es identico al de hoy. El
sistema solo se desvia de los defaults cuando el learner escribio valores probados.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path("data")
_ACTIVE_FILE = _DATA_DIR / "learned_params.json"


@dataclass(frozen=True)
class TunableParam:
    """Definicion de un parametro que el learner puede variar.

    name:    clave estable usada en el JSON y por las funciones de señal.
    default: valor de fabrica (== comportamiento actual del sistema).
    lo/hi:   limites duros; el learner nunca propone fuera de [lo, hi].
    scale:   desviacion tipica de la perturbacion a temperatura 1.0.
    is_int:  si el valor debe redondearse a entero (ej. lookbacks de velas).
    setups:  setups cuya performance depende de este parametro. Cuando uno de
             esos setups prueba edge negativo, el learner prioriza variar este
             parametro (asignacion de credito).
    help:    descripcion corta para el dashboard.
    """
    name: str
    default: float
    lo: float
    hi: float
    scale: float
    is_int: bool = False
    setups: tuple[str, ...] = field(default_factory=tuple)
    help: str = ""

    def clamp(self, value: float) -> float:
        v = max(self.lo, min(self.hi, float(value)))
        return float(round(v)) if self.is_int else round(v, 5)


# ── Registro de parametros tuneables ──────────────────────────────────────────
# Curado: los parametros de mayor impacto que se pueden leer en runtime de forma
# segura. Cada uno mapea a los setups que afecta para asignacion de credito.

TUNABLES: dict[str, TunableParam] = {
    "smc_eql_tolerance": TunableParam(
        name="smc_eql_tolerance", default=0.005, lo=0.002, hi=0.012, scale=0.003,
        setups=("STOCK_SMC_REVERSAL", "STOCK_SWING_TREND_PULLBACK"),
        help="Tolerancia para agrupar Equal Highs/Lows (niveles de liquidez SMC).",
    ),
    "smc_pda_lookback": TunableParam(
        name="smc_pda_lookback", default=60, lo=30, hi=120, scale=18, is_int=True,
        setups=("STOCK_SWING_TREND_PULLBACK", "STOCK_SMC_REVERSAL"),
        help="Velas de lookback para el swing de referencia del PDA array (zona premium/discount).",
    ),
    "stock_min_net_rr": TunableParam(
        name="stock_min_net_rr", default=0.4, lo=0.25, hi=0.9, scale=0.12,
        setups=("STOCK_SWING_BREAKOUT", "STOCK_SWING_TREND_PULLBACK", "STOCK_SMC_REVERSAL"),
        help="R/R minimo NETO de costos para emitir una señal de accion.",
    ),
    "stock_max_breakout_extension": TunableParam(
        name="stock_max_breakout_extension", default=0.05, lo=0.02, hi=0.10, scale=0.02,
        setups=("STOCK_SWING_BREAKOUT",),
        help="Extension maxima sobre el pivote para un breakout no tardio (regla Minervini).",
    ),
    "stock_min_gross_rr": TunableParam(
        name="stock_min_gross_rr", default=1.2, lo=1.0, hi=2.5, scale=0.25,
        setups=("STOCK_SWING_BREAKOUT", "STOCK_SWING_TREND_PULLBACK", "STOCK_SMC_REVERSAL"),
        help="R/R bruto minimo antes de costos; evita trades con recorrido insuficiente.",
    ),
    "stock_pullback_keltner_atr_mult": TunableParam(
        name="stock_pullback_keltner_atr_mult", default=2.0, lo=1.2, hi=3.2, scale=0.35,
        setups=("STOCK_SWING_TREND_PULLBACK",),
        help="Multiplicador ATR de la banda Keltner usada para detectar pullbacks operables.",
    ),
    "stock_pullback_rsi_min": TunableParam(
        name="stock_pullback_rsi_min", default=30.0, lo=20.0, hi=45.0, scale=4.0,
        setups=("STOCK_SWING_TREND_PULLBACK",),
        help="RSI minimo para evitar comprar caida libre en pullbacks.",
    ),
    "stock_pullback_rsi_max": TunableParam(
        name="stock_pullback_rsi_max", default=65.0, lo=55.0, hi=75.0, scale=4.0,
        setups=("STOCK_SWING_TREND_PULLBACK",),
        help="RSI maximo para no comprar pullbacks ya recalentados.",
    ),
    "stock_breakout_volume_mult": TunableParam(
        name="stock_breakout_volume_mult", default=1.10, lo=1.0, hi=2.0, scale=0.18,
        setups=("STOCK_SWING_BREAKOUT",),
        help="Volumen minimo vs promedio para validar breakout en BYMA.",
    ),
    "smc_reversal_stop_atr_mult": TunableParam(
        name="smc_reversal_stop_atr_mult", default=2.5, lo=1.2, hi=4.0, scale=0.45,
        setups=("STOCK_SMC_REVERSAL",),
        help="Distancia maxima del stop SMC medida en ATR.",
    ),
    "smc_reversal_target_atr_mult": TunableParam(
        name="smc_reversal_target_atr_mult", default=2.0, lo=1.2, hi=4.5, scale=0.45,
        setups=("STOCK_SMC_REVERSAL",),
        help="Target fallback SMC cuando no hay liquidez BSL limpia arriba.",
    ),
    # ── Volumen (volume_momentum.py) — detectar la ola temprano en BYMA ─────────
    "rvol_explosive_threshold": TunableParam(
        name="rvol_explosive_threshold", default=2.5, lo=1.5, hi=4.0, scale=0.4,
        setups=("STOCK_SWING_BREAKOUT", "STOCK_SMC_REVERSAL"),
        help="RVOL minimo para flujo institucional (ola temprana). >2.5x = alguien grande compra.",
    ),
    "rvol_high_threshold": TunableParam(
        name="rvol_high_threshold", default=1.5, lo=1.1, hi=2.5, scale=0.25,
        setups=("STOCK_SWING_BREAKOUT", "STOCK_SWING_TREND_PULLBACK", "STOCK_SMC_REVERSAL"),
        help="RVOL minimo para participacion alta (ola arrancando).",
    ),
    "rvol_lookback": TunableParam(
        name="rvol_lookback", default=20, lo=10, hi=40, scale=5, is_int=True,
        setups=("STOCK_SWING_BREAKOUT", "STOCK_SWING_TREND_PULLBACK", "STOCK_SMC_REVERSAL"),
        help="Ruedas para el volumen promedio de referencia del RVOL.",
    ),
}


# ── Flags de estrategia (que setups estan activos) ───────────────────────────
# El tree-search los enciende/apaga para descubrir QUE estrategia opera mejor,
# no solo como tunear cada parametro. Default: todos activos (== hoy).

FLAGS: dict[str, bool] = {
    "enable_smc_reversal": True,    # stop-hunt SSL + reversion (sniper)
    "enable_swing_pullback": True,  # pullback a Keltner en tendencia
    "enable_swing_breakout": True,  # breakout Minervini con volumen
}


# ── Override en memoria (para evaluar genomas sin tocar el archivo en vivo) ────
# El backtester prueba miles de configs; no queremos escribir disco por cada una.
# `param_override` consulta antes que el archivo y se restaura al salir del with.

_override: Optional[dict[str, float]] = None
_override_flags: Optional[dict[str, bool]] = None


class param_override:
    """Context manager: fuerza un set de params/flags en memoria temporalmente.

    Uso:
        with param_override({"stock_min_net_rr": 0.6}, {"enable_swing_breakout": False}):
            señales = scan_stock_symbol(...)   # ve esos valores
    """

    def __init__(
        self,
        params: Optional[dict[str, float]] = None,
        flags: Optional[dict[str, bool]] = None,
    ) -> None:
        self._params = params
        self._flags = flags
        self._prev_p = None
        self._prev_f = None

    def __enter__(self):
        global _override, _override_flags
        self._prev_p, self._prev_f = _override, _override_flags
        _override = dict(self._params) if self._params is not None else None
        _override_flags = dict(self._flags) if self._flags is not None else None
        return self

    def __exit__(self, *exc):
        global _override, _override_flags
        _override, _override_flags = self._prev_p, self._prev_f
        return False


# ── Cache del set activo (re-lee si cambia el archivo en disco) ───────────────

_cache: dict[str, float] = {}
_cache_mtime: Optional[float] = None
_cache_path: Optional[Path] = None
_flags_cache: dict[str, bool] = {}
_flags_cache_mtime: Optional[float] = None


def _active_path(data_dir: Optional[Path] = None) -> Path:
    return Path(data_dir) / "learned_params.json" if data_dir else _ACTIVE_FILE


def active_params(data_dir: Optional[Path] = None) -> dict[str, float]:
    """Devuelve el set de parametros activo desde disco (cacheado por mtime).

    Vacio si no existe el archivo → los callers usan sus defaults.
    """
    global _cache, _cache_mtime, _cache_path
    path = _active_path(data_dir)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        # Sin archivo → set vacio. Invalida cache previa.
        _cache, _cache_mtime, _cache_path = {}, None, path
        return {}

    if _cache_path == path and _cache_mtime == mtime:
        return _cache

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        params = raw.get("params", {}) if isinstance(raw, dict) else {}
        # Solo conservar claves conocidas y clampeadas a sus limites.
        clean: dict[str, float] = {}
        for k, v in params.items():
            spec = TUNABLES.get(k)
            if spec is not None and isinstance(v, (int, float)):
                clean[k] = spec.clamp(v)
        _cache, _cache_mtime, _cache_path = clean, mtime, path
        return clean
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        logger.debug("learned_params ilegible: %s", exc)
        _cache, _cache_mtime, _cache_path = {}, mtime, path
        return {}


def param(name: str, default: float, data_dir: Optional[Path] = None) -> float:
    """Lee un parametro activo; si no esta aprendido, devuelve `default`.

    Prioridad: override en memoria (backtester) → archivo en disco → default.
    Backward-compatible: sin override ni archivo, devuelve el default.
    """
    if _override is not None and name in _override:
        spec = TUNABLES.get(name)
        return spec.clamp(_override[name]) if spec else _override[name]
    return active_params(data_dir).get(name, default)


def active_flags(data_dir: Optional[Path] = None) -> dict[str, bool]:
    """Flags de setups activos desde disco (cacheado por mtime). Vacio → defaults."""
    global _flags_cache, _flags_cache_mtime
    path = _active_path(data_dir)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _flags_cache, _flags_cache_mtime = {}, None
        return {}
    if _flags_cache_mtime == mtime:
        return _flags_cache
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        flags = raw.get("flags", {}) if isinstance(raw, dict) else {}
        clean = {k: bool(v) for k, v in flags.items() if k in FLAGS}
        _flags_cache, _flags_cache_mtime = clean, mtime
        return clean
    except (OSError, json.JSONDecodeError, AttributeError):
        _flags_cache, _flags_cache_mtime = {}, mtime
        return {}


def flag(name: str, default: Optional[bool] = None, data_dir: Optional[Path] = None) -> bool:
    """Lee un flag de estrategia (override en memoria → archivo → default del registro)."""
    if _override_flags is not None and name in _override_flags:
        return bool(_override_flags[name])
    base = FLAGS.get(name, True if default is None else default)
    return bool(active_flags(data_dir).get(name, base))


def write_active_params(
    params: dict[str, float],
    *,
    flags: Optional[dict[str, bool]] = None,
    data_dir: Optional[Path] = None,
    source: str = "learner",
) -> None:
    """Persiste el set activo de parametros (y flags) — lo despliega para las señales."""
    from datetime import datetime
    path = _active_path(data_dir)
    clean = {
        k: TUNABLES[k].clamp(v)
        for k, v in params.items()
        if k in TUNABLES and isinstance(v, (int, float))
    }
    payload = {
        "params": clean,
        "flags": {k: bool(v) for k, v in (flags or {}).items() if k in FLAGS},
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        # Bust cache para que el proximo active_params/active_flags relea.
        global _cache_mtime, _flags_cache_mtime
        _cache_mtime = None
        _flags_cache_mtime = None
    except OSError as exc:
        logger.debug("no se pudo escribir learned_params: %s", exc)


def default_flags() -> dict[str, bool]:
    """Flags de fabrica (todos los setups activos)."""
    return dict(FLAGS)


def default_params() -> dict[str, float]:
    """El set de parametros de fabrica (punto de partida del campeon)."""
    return {name: spec.default for name, spec in TUNABLES.items()}


def describe_active(data_dir: Optional[Path] = None) -> list[dict]:
    """Para el dashboard: cada param con su valor activo vs default y limites."""
    active = active_params(data_dir)
    rows = []
    for name, spec in TUNABLES.items():
        cur = active.get(name, spec.default)
        rows.append({
            "param": name,
            "actual": cur,
            "default": spec.default,
            "min": spec.lo,
            "max": spec.hi,
            "delta_vs_default": round(cur - spec.default, 5),
            "help": spec.help,
        })
    return rows
