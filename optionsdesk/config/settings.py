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


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Primary Trading API / Matriz OMS. El host depende del ALyC:
    # p.ej. https://api.remarkets.primary.com.ar para sandbox.
    primary_base_url: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_BASE_URL", "")
    )
    primary_ws_url: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_WS_URL", "")
    )
    primary_user: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_USER", "")
    )
    primary_password: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_PASSWORD", "")
    )
    primary_spot_symbol: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_SPOT_SYMBOL", "")
    )
    primary_caucion_symbol: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_CAUCION_SYMBOL", "")
    )
    primary_account_name: str = field(
        default_factory=lambda: os.environ.get("PRIMARY_ACCOUNT_NAME", "")
    )
    primary_ws_level: int = field(
        default_factory=lambda: min(max(_env_int("PRIMARY_WS_LEVEL", 1), 1), 2)
    )
    primary_ws_depth: int = field(
        default_factory=lambda: min(max(_env_int("PRIMARY_WS_DEPTH", 1), 1), 5)
    )
    primary_warmup_s: int = field(
        default_factory=lambda: max(_env_int("PRIMARY_WARMUP_S", 60), 0)
    )
    primary_shadow_complete: bool = field(
        default_factory=lambda: _env_bool("PRIMARY_SHADOW_COMPLETE", False)
    )

    # Tickets reales bloqueados hasta confirmar el tarifario efectivo del ALyC.
    costs_verified: bool = field(
        default_factory=lambda: os.environ.get("COSTS_VERIFIED", "false").lower() == "true"
    )
    costs_profile: str = field(
        default_factory=lambda: os.environ.get("COSTS_PROFILE", "conservative-unverified")
    )

    # InvertirOnLine (IOL) credentials — proveedor primario v2.8+
    iol_user: str = field(default_factory=lambda: os.environ.get("IOL_USER", ""))
    iol_password: str = field(default_factory=lambda: os.environ.get("IOL_PASSWORD", ""))
    # IOL es REST y bonifica una cantidad mensual limitada de API calls.
    # Primary WebSocket sigue siendo la via recomendada para el pulso intradia.
    iol_dashboard_refresh_s: int = field(
        default_factory=lambda: _env_int("IOL_DASHBOARD_REFRESH_S", 60)
    )
    iol_recorder_interval_s: int = field(
        default_factory=lambda: _env_int("IOL_RECORDER_INTERVAL_S", 900)
    )
    iol_option_quotes_per_cycle: int = field(
        default_factory=lambda: _env_int("IOL_OPTION_QUOTES_PER_CYCLE", 8)
    )
    iol_options_list_ttl_s: int = field(
        default_factory=lambda: _env_int("IOL_OPTIONS_LIST_TTL_S", 900)
    )
    iol_provisional_tickets_enabled: bool = field(
        default_factory=lambda: _env_bool("IOL_PROVISIONAL_TICKETS_ENABLED", True)
    )
    iol_ticket_ttl_s: int = field(
        default_factory=lambda: _env_int("IOL_TICKET_TTL_S", 60)
    )
    iol_ticket_max_quote_age_s: int = field(
        default_factory=lambda: _env_int("IOL_TICKET_MAX_QUOTE_AGE_S", 75)
    )
    iol_ticket_max_spread_pct: float = field(
        default_factory=lambda: _env_float("IOL_TICKET_MAX_SPREAD_PCT", 6.0) or 6.0
    )
    spot_tape_interval_s: int = field(
        default_factory=lambda: _env_int("SPOT_TAPE_INTERVAL_S", 60)
    )

    # Acciones argentinas: panel long-only con demo automatico read-only.
    stock_universe: str = field(
        default_factory=lambda: os.environ.get(
            "STOCK_UNIVERSE",
            "GGAL,YPFD,PAMP,BBAR,SUPV,COME,ALUA,TXAR,CEPU,EDN,TGSU2,TRAN,LOMA,BYMA,VALO,MIRG",
        )
    )
    stock_demo_trade_ars: float = field(
        default_factory=lambda: _env_float("STOCK_DEMO_TRADE_ARS", 100_000.0) or 100_000.0
    )
    stock_demo_max_scalps: int = field(
        default_factory=lambda: _env_int("STOCK_DEMO_MAX_SCALPS", 0)
    )
    stock_demo_max_swings: int = field(
        default_factory=lambda: _env_int("STOCK_DEMO_MAX_SWINGS", 5)
    )
    stock_demo_quote_max_age_s: int = field(
        default_factory=lambda: _env_int("STOCK_DEMO_QUOTE_MAX_AGE_S", 90)
    )
    stock_demo_refresh_interval_s: int = field(
        default_factory=lambda: _env_int("STOCK_DEMO_REFRESH_INTERVAL_S", 60)
    )
    # Auto-aprendizaje del runner: baseline (ignora trades anteriores a esta fecha ISO
    # — útil para resetear tras deshabilitar una estrategia) y ventana de drawdown.
    demo_learning_since: Optional[str] = field(
        default_factory=lambda: os.environ.get("DEMO_LEARNING_SINCE") or None
    )
    demo_learning_lookback: int = field(
        default_factory=lambda: _env_int("DEMO_LEARNING_LOOKBACK", 40)
    )
    circuit_breaker_consecutive_losses: int = field(
        default_factory=lambda: _env_int("CIRCUIT_BREAKER_CONSECUTIVE_LOSSES", 3)
    )
    circuit_breaker_drawdown_pct: float = field(
        default_factory=lambda: _env_float("CIRCUIT_BREAKER_DRAWDOWN_PCT", 5.0) or 5.0
    )
    stock_max_spread_pct: float = field(
        default_factory=lambda: _env_float("STOCK_MAX_SPREAD_PCT", 2.5) or 2.5
    )
    stock_min_daily_volume: int = field(
        default_factory=lambda: _env_int("STOCK_MIN_DAILY_VOLUME", 50000)
    )
    stock_demo_positions_file: Path = field(
        default_factory=lambda: Path(os.environ.get("STOCK_DEMO_POSITIONS_FILE", "data/stock_demo_positions.json"))
    )
    stock_demo_trades_file: Path = field(
        default_factory=lambda: Path(os.environ.get("STOCK_DEMO_TRADES_FILE", "data/stock_demo_trades.jsonl"))
    )
    stock_tape_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("STOCK_TAPE_DIR", "data/stock_tape"))
    )

    # HomeBroker legacy: retenido solo para compatibilidad, fuera del arranque operativo.
    hb_dni: str = field(default_factory=lambda: os.environ.get("HB_DNI", ""))
    hb_user: str = field(default_factory=lambda: os.environ.get("HB_USER", ""))
    hb_password: str = field(default_factory=lambda: os.environ.get("HB_PASSWORD", ""))
    hb_broker_id: int = field(
        default_factory=lambda: _env_int("HB_BROKER_ID", 0)
    )
    hb_settlement: str = field(
        default_factory=lambda: os.environ.get("HB_SETTLEMENT", "24hs")
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

    # Recorder snapshot interval (seconds). Primary usa streaming.
    # El recorder aplica un piso propio mas alto cuando el fallback es IOL REST.
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
    manual_reconciliation_file: Path = field(
        default_factory=lambda: Path(os.environ.get("MANUAL_RECONCILIATION_FILE", "data/manual_reconciliation.json"))
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
        default_factory=lambda: _env_float("SCALPING_RISK_PER_TRADE_PCT", 0.35) or 0.35
    )
    scalping_max_open_positions: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_OPEN_POSITIONS", 1)
    )
    scalping_max_total_risk_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_MAX_TOTAL_RISK_PCT", 0.35) or 0.35
    )
    scalping_no_progress_min: int = field(
        default_factory=lambda: _env_int("SCALPING_NO_PROGRESS_MIN", 20)
    )
    scalping_flat_before_close_min: int = field(
        default_factory=lambda: _env_int("SCALPING_FLAT_BEFORE_CLOSE_MIN", 15)
    )
    scalping_start_after_open_min: int = field(
        default_factory=lambda: _env_int("SCALPING_START_AFTER_OPEN_MIN", 20)
    )
    scalping_ticket_ttl_s: int = field(
        default_factory=lambda: _env_int("SCALPING_TICKET_TTL_S", 15)
    )
    scalping_spot_max_age_s: int = field(
        default_factory=lambda: _env_int("SCALPING_SPOT_MAX_AGE_S", 5)
    )
    scalping_ws_max_age_s: int = field(
        default_factory=lambda: _env_int("SCALPING_WS_MAX_AGE_S", 5)
    )
    scalping_contract_max_age_s: int = field(
        default_factory=lambda: _env_int("SCALPING_CONTRACT_MAX_AGE_S", 30)
    )
    scalping_initial_lot_cap: int = field(
        default_factory=lambda: _env_int("SCALPING_INITIAL_LOT_CAP", 2)
    )
    scalping_manual_sizing: bool = field(
        default_factory=lambda: _env_bool("SCALPING_MANUAL_SIZING", True)
    )
    scalping_manual_max_lots: int = field(
        default_factory=lambda: max(_env_int("SCALPING_MANUAL_MAX_LOTS", 10), 1)
    )
    scalping_daily_loss_pct: float = field(
        default_factory=lambda: _env_float("SCALPING_DAILY_LOSS_PCT", 1.5) or 1.5
    )
    scalping_max_stops_per_day: int = field(
        default_factory=lambda: _env_int("SCALPING_MAX_STOPS_PER_DAY", 2)
    )
    # Ventana antes del cierre para evaluar overnight solo cuando el operador
    # lo habilita explicitamente: DTE>=5, filtros estrictos y sizing reducido.
    scalping_eod_window_min: int = field(
        default_factory=lambda: _env_int("SCALPING_EOD_WINDOW_MIN", 35)
    )

    # ── Smart Money Concepts (SMC) ────────────────────────────────────────────
    smc_enabled: bool = field(
        default_factory=lambda: _env_bool("SMC_ENABLED", True)
    )
    # Armónicos M/W — Fase 3 (harmonics.py)
    harmonics_enabled: bool = field(
        default_factory=lambda: _env_bool("HARMONICS_ENABLED", False)
    )
    # ADR/CCL confluence — Fase 2 (providers/adr.py)
    adr_ccl_enabled: bool = field(
        default_factory=lambda: _env_bool("ADR_CCL_ENABLED", False)
    )
    # EMAs del Market Maker Method (vol 2)
    mm_ema_periods: tuple = field(
        default_factory=lambda: (5, 13, 50, 200)
    )
    # Mapa ticker local → ADR en NYSE y ratio (acciones locales por ADS).
    # Formato env: "GGAL:GGAL.US:10,YPFD:YPF.US:1,..." — VERIFICAR ratios reales.
    # Ratios verificados contra prospectos SEC/CNV (acciones locales por 1 ADS).
    # GGAL=10, YPF=1, PAM=25, SUPV=5, CEPU=2, EDN=20, TGS=5, LOMA=5, BMA=10.
    # BBAR (BBVA Argentina) NO tiene ADR activo en NYSE/Nasdaq — excluido.
    # Si tenés otros ratios confirmados podés overridear via ADR_MAP en el .env.
    adr_map_raw: str = field(
        default_factory=lambda: os.environ.get(
            "ADR_MAP",
            "GGAL:GGAL.US:10,YPFD:YPF.US:1,PAMP:PAM.US:25,"
            "SUPV:SUPV.US:5,CEPU:CEPU.US:2,EDN:EDN.US:20,"
            "TGSU2:TGS.US:5,LOMA:LOMA.US:5,BMA:BMA.US:10",
        )
    )
    # Stooq API key opcional para histórico masivo (backfill de ADR diario).
    # Sin key: solo endpoint liviano /q/l/ (cotización actual).
    stooq_apikey: str = field(
        default_factory=lambda: os.environ.get("STOOQ_APIKEY", "")
    )

    def adr_map(self) -> dict[str, tuple[str, float]]:
        """Parsea ADR_MAP en {ticker_local: (ticker_adr, ratio)}."""
        result: dict[str, tuple[str, float]] = {}
        for entry in self.adr_map_raw.split(","):
            parts = entry.strip().split(":")
            if len(parts) == 3:
                local, adr_ticker, ratio_str = parts
                try:
                    result[local.strip().upper()] = (adr_ticker.strip(), float(ratio_str))
                except ValueError:
                    pass
        return result

    def is_iol_configured(self) -> bool:
        return bool(self.iol_user and self.iol_password)

    def is_primary_configured(self) -> bool:
        return bool(self.primary_base_url and self.primary_user and self.primary_password)

    def is_configured(self) -> bool:
        """Legacy: True si hay credenciales de Bull Market Homebroker."""
        return bool(self.hb_dni and self.hb_user and self.hb_password and self.hb_broker_id)

    def stock_universe_symbols(self) -> list[str]:
        raw = self.stock_universe.replace(";", ",")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
        return list(dict.fromkeys(symbols))


settings = Settings()
