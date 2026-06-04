"""Interfaz abstracta para todas las fuentes de datos de mercado."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


@dataclass
class Quote:
    """Quote de mercado para un instrumento."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    bid_size: float = 0.0
    ask_size: float = 0.0
    source: str = ""
    received_at: Optional[datetime] = None
    market_ts: Optional[datetime] = None
    book_ts: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.received_at is None:
            self.received_at = self.timestamp

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if m <= 0:
            return 0.0
        return (self.ask - self.bid) / m * 100.0

    def is_stale(self, max_age_s: float = 300.0) -> bool:
        freshness_ts = self.book_ts or self.received_at or self.timestamp
        now = datetime.now(freshness_ts.tzinfo) if freshness_ts.tzinfo else datetime.now()
        dt = (now - freshness_ts).total_seconds()
        return dt > max_age_s


@dataclass
class OptionsChain:
    """Snapshot completo de la cadena de opciones de GGAL."""

    underlying: str
    spot: Quote
    options: dict[str, Quote]
    timestamp: datetime = field(default_factory=datetime.now)
    expiry_calendar: dict[str, date] = field(default_factory=dict)
    min_price_increments: dict[str, float] = field(default_factory=dict)


@dataclass
class MarketDataHealth:
    """Estado operativo del feed de mercado para diagnostico y UI."""

    source: str
    connected: bool
    last_success_ts: Optional[datetime] = None
    last_error: str = ""
    last_latency_ms: Optional[float] = None
    requests: int = 0
    timeouts: int = 0
    retries: int = 0
    options_seen: int = 0
    options_tradeable: int = 0
    ticket_capable: bool = False
    ws_age_s: Optional[float] = None
    warmup_complete: bool = False
    coverage_pct: float = 0.0
    blocking_reason: str = ""
    operational_mode: str = "BLOCKED"


@dataclass
class AccountSnapshot:
    """Read-only account view. Order routing deliberately stays outside this app."""

    account_name: str
    source: str
    fetched_at: datetime
    positions: list[dict[str, Any]] = field(default_factory=list)
    available: bool = False
    error: str = ""

    @property
    def open_positions(self) -> int:
        return len(self.positions)


class MarketDataProvider(ABC):
    """Interfaz para todas las fuentes de datos de mercado."""

    @abstractmethod
    def get_options_chain(self) -> Optional[OptionsChain]:
        """Cadena de opciones GGAL mas reciente. None si no hay datos."""
        ...

    @abstractmethod
    def get_spot(self) -> Optional[Quote]:
        """Quote de la accion GGAL."""
        ...

    def get_quote(self, symbol: str) -> Optional[Quote]:
        """Quote de una accion argentina. Por compatibilidad, default solo GGAL."""
        if str(symbol).upper() == "GGAL":
            return self.get_spot()
        return None

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Quotes por simbolo. Providers concretos pueden optimizar batching/paralelismo."""
        quotes: dict[str, Quote] = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote is not None:
                quotes[quote.symbol.upper()] = quote
        return quotes

    @abstractmethod
    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        """TNA (%) de la caucion colocadora para el plazo dado."""
        ...

    def is_connected(self) -> bool:
        return False

    def get_health(self) -> MarketDataHealth:
        return MarketDataHealth(source=self.__class__.__name__, connected=self.is_connected())

    def get_account_snapshot(self, account_name: str = "") -> AccountSnapshot:
        return AccountSnapshot(
            account_name=account_name,
            source=self.__class__.__name__,
            fetched_at=datetime.now(),
            error="Risk API no disponible para este proveedor",
        )
