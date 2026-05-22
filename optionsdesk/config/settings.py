from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Bull Market homebroker credentials
    hb_dni: str = field(default_factory=lambda: os.environ.get("HB_DNI", ""))
    hb_user: str = field(default_factory=lambda: os.environ.get("HB_USER", ""))
    hb_password: str = field(default_factory=lambda: os.environ.get("HB_PASSWORD", ""))
    hb_broker_id: int = field(
        default_factory=lambda: int(os.environ.get("HB_BROKER_ID", "0"))
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
        default_factory=lambda: float(os.environ.get("MIN_TNA_SPREAD_PCT", "5.0"))
    )

    # Recommender / dashboard
    directional_enabled: bool = field(
        default_factory=lambda: os.environ.get("DIRECTIONAL_ENABLED", "false").lower() == "true"
    )
    default_capital: Optional[float] = field(
        default_factory=lambda: (
            float(os.environ.get("DEFAULT_CAPITAL", "0")) or None
        )
    )

    # Market schedule
    market_tz: str = "America/Argentina/Buenos_Aires"
    market_open: str = "10:30"
    market_close: str = "17:00"

    # Recorder snapshot interval (seconds)
    recorder_interval_s: int = field(
        default_factory=lambda: int(os.environ.get("RECORDER_INTERVAL_S", "120"))
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
        default_factory=lambda: float(os.environ.get("SWING_SPREAD_BUYBACK_PCT", "0.10"))
    )
    open_positions_file: Path = field(
        default_factory=lambda: Path(os.environ.get("OPEN_POSITIONS_FILE", "data/open_positions.jsonl"))
    )

    def is_configured(self) -> bool:
        return bool(self.hb_dni and self.hb_user and self.hb_password and self.hb_broker_id)


settings = Settings()
