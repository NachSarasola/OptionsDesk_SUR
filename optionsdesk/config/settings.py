from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float = 0.0) -> Optional[float]:
    """Lee una variable de entorno como float; ignora valores no numéricos (p.ej. comentarios)."""
    raw = os.environ.get(key, "")
    if not raw:
        v = float(default)
        return v if v != 0.0 else None
    try:
        val = float(raw)
        return val if val != 0.0 else None
    except (ValueError, TypeError):
        v = float(default)
        return v if v != 0.0 else None


def _env_int(key: str, default: int = 0) -> int:
    """Lee una variable de entorno como int; devuelve default si no es numérico."""
    raw = os.environ.get(key, "")
    try:
        return int(raw) if raw else default
    except (ValueError, TypeError):
        return default


@dataclass
class Settings:
    # InvertirOnLine (IOL) credentials — proveedor primario v2.8+
    iol_user: str = field(default_factory=lambda: os.environ.get("IOL_USER", ""))
    iol_password: str = field(default_factory=lambda: os.environ.get("IOL_PASSWORD", ""))

    # Bull Market homebroker credentials (legacy — fallback si no hay IOL)
    hb_dni: str = field(default_factory=lambda: os.environ.get("HB_DNI", ""))
    hb_user: str = field(default_factory=lambda: os.environ.get("HB_USER", ""))
    hb_password: str = field(default_factory=lambda: os.environ.get("HB_PASSWORD", ""))
    hb_broker_id: int = field(
        default_factory=lambda: _env_int("HB_BROKER_ID", 0)
    )

    # Telegram alerts
    telegram_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_TOKEN", "")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", "")
    )

    # Strategy thresholds
    min_tna_spread_pct: float = field(
        default_factory=lambda: _env_float("MIN_TNA_SPREAD_PCT", 5.0) or 5.0
    )

    # Recommender / dashboard
    directional_enabled: bool = field(
        default_factory=lambda: os.environ.get("DIRECTIONAL_ENABLED", "false").lower() == "true"
    )
    default_capital: Optional[float] = field(
        default_factory=lambda: _env_float("DEFAULT_CAPITAL", 0.0)
    )
    
    # Fallback risk-free rate
    default_caucion_tna: float = field(
        default_factory=lambda: _env_float("DEFAULT_CAUCION_TNA", 45.0) or 45.0
    )

    # Market schedule
    market_tz: str = "America/Argentina/Buenos_Aires"
    market_open: str = "10:30"
    market_close: str = "17:00"

    # Recorder snapshot interval (seconds). Default 15s: pyhomebroker usa WebSocket
    # streaming, los datos llegan en segundos. 120s era excesivamente conservador.
    recorder_interval_s: int = field(
        default_factory=lambda: _env_int("RECORDER_INTERVAL_S", 15)
    )

    # Snapshot storage
    snapshots_dir: Path = field(default_factory=lambda: Path("data/snapshots"))

    # Historial del subyacente (BYMA Open Data vía PyOBD)
    history_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("HISTORY_DIR", "data/history"))
    )

    # Horizon optimizer / monitor de posiciones
    horizon_monitor_enabled: bool = field(
        default_factory=lambda: os.environ.get("HORIZON_MONITOR_ENABLED", "false").lower() == "true"
    )
    swing_spread_buyback_pct: float = field(
        default_factory=lambda: _env_float("SWING_SPREAD_BUYBACK_PCT", 0.10) or 0.10
    )
    open_positions_file: Path = field(
        default_factory=lambda: Path(os.environ.get("OPEN_POSITIONS_FILE", "data/open_positions.jsonl"))
    )

    # Edge cuantitativo (v2.3+)
    events_enabled: bool = field(
        default_factory=lambda: os.environ.get("EVENTS_ENABLED", "true").lower() == "true"
    )
    # VRP mínimo para gate blando (fracción, p.ej. -0.05 rechaza VRP < -5pp).
    # Por default -999 = sin gate duro (solo informativo).
    vrp_min_threshold: float = field(
        default_factory=lambda: _env_float("VRP_MIN_THRESHOLD", -999.0) or -999.0
    )
    # Umbral de captura default para take-profit (%)
    default_target_capture_pct: float = field(
        default_factory=lambda: _env_float("DEFAULT_TARGET_CAPTURE_PCT", 50.0) or 50.0
    )
    # Días a vencimiento para señal de roll
    default_roll_dte: int = field(
        default_factory=lambda: _env_int("DEFAULT_ROLL_DTE", 21)
    )
    # Múltiplo de prima para stop por pérdida
    default_max_loss_mult: float = field(
        default_factory=lambda: _env_float("DEFAULT_MAX_LOSS_MULT", 2.0) or 2.0
    )

    # Scalping module
    scalping_min_volume: int = field(
        default_factory=lambda: _env_int("SCALPING_MIN_VOLUME", 1)
    )
    scalping_max_spread_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_MAX_SPREAD_PCT", 20.0) or 20.0
    )
    scalping_momentum_threshold: float = field(
        default_factory=lambda: _env_float("SCALPING_MOMENTUM_THRESHOLD", 1.0) or 1.0
    )
    scalping_mispricing_min_pp: float = field(
        default_factory=lambda: _env_float("SCALPING_MISPRICING_MIN_PP", 2.0) or 2.0
    )
    scalping_min_rr: float = field(
        default_factory=lambda: _env_float("SCALPING_MIN_RR", 1.2) or 1.2
    )
    scalping_max_quote_age_s: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_QUOTE_AGE_S", 90)
    )
    scalping_risk_per_trade_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_RISK_PER_TRADE_PCT", 1.0) or 1.0
    )
    scalping_max_open_positions: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_OPEN_POSITIONS", 3)
    )
    scalping_max_total_risk_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_MAX_TOTAL_RISK_PCT", 4.0) or 4.0
    )
    scalping_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_DAILY_LOSS_PCT", 2.0) or 2.0
    )
    scalping_max_trades_per_day: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_TRADES_PER_DAY", 5)
    )
    scalping_cooldown_after_losses: int = field(
        default_factory=lambda: _env_int("SCALPING_COOLDOWN_AFTER_LOSSES", 2)
    )
    scalping_max_hold_min: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_HOLD_MIN", 90)
    )
    scalping_no_progress_min: int = field(
        default_factory=lambda: _env_int("SCALPING_NO_PROGRESS_MIN", 20)
    )
    scalping_flat_before_close_min: int = field(
        default_factory=lambda: _env_int("SCALPING_FLAT_BEFORE_CLOSE_MIN", 10)
    )

    def is_iol_configured(self) -> bool:
        return bool(self.iol_user and self.iol_password)

    def is_configured(self) -> bool:
        """Legacy: True si hay credenciales de Bull Market Homebroker."""
        return bool(self.hb_dni and self.hb_user and self.hb_password and self.hb_broker_id)


settings = Settings()
