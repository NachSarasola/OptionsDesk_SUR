"""Read-only market data provider for Primary Trading API / Matriz OMS.

Primary documents REST for authentication and instrument discovery, and
WebSocket subscriptions for real-time market data. This adapter deliberately
does not expose order routing.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional
from urllib.parse import quote as urlquote
from urllib.parse import urlsplit, urlunsplit
from zoneinfo import ZoneInfo

import requests

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import (
    AccountSnapshot,
    MarketDataHealth,
    MarketDataProvider,
    OptionsChain,
    Quote,
)

logger = logging.getLogger(__name__)

_AUTH_PATH = "auth/getToken"
_DETAILS_PATH = "rest/instruments/details"
_ALL_INSTRUMENTS_PATH = "rest/instruments/all"
_RETRIES = 2
_RETRY_BASE_SLEEP_S = 0.30
_SUBSCRIPTION_BATCH_SIZE = 1000
_RECONNECT_COOLDOWN_S = 30.0
_CHAIN_WAIT_S = 3.0
_WS_ENTRIES = ("BI", "OF", "LA", "CL", "ACP", "NV", "TV")
_OPTION_TOKEN_RE = re.compile(r"(?<![A-Z0-9])(GFG[CV]\d+(?:[.,]\d+)?[A-Z]+)(?![A-Z0-9])", re.I)
_GGAL_TOKEN_RE = re.compile(r"(?<![A-Z0-9])GGAL(?![A-Z0-9])", re.I)
_EXPIRY_CODE_RE = re.compile(r"([A-Z]+)$")


def _derive_ws_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    scheme = "wss" if parts.scheme == "https" else "ws"
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _today_ba() -> date:
    return datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).date()


def _is_byma_market_session() -> bool:
    now = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    minutes = now.hour * 60 + now.minute
    return now.weekday() < 5 and (11 * 60) <= minutes < (17 * 60)


def _canonical_option_symbol(symbol: str) -> Optional[str]:
    match = _OPTION_TOKEN_RE.search(symbol.strip())
    return match.group(1).upper().replace(",", ".") if match else None


def _parse_maturity(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    raw = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw[:10], fmt).date()
        except ValueError:
            continue
    return None


def _entry_number(value: Any, *fields: str) -> float:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        for field in fields:
            if value.get(field) is not None:
                value = value[field]
                break
        else:
            return 0.0
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _market_data_latency_ms(market_data: dict[str, Any]) -> Optional[float]:
    epochs: list[float] = []
    for value in market_data.values():
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            if not isinstance(item, dict) or item.get("date") is None:
                continue
            try:
                epochs.append(float(item["date"]))
            except (TypeError, ValueError):
                pass
    if not epochs:
        return None
    return round(max((time.time() * 1000.0) - max(epochs), 0.0), 2)


def _market_timestamp(market_data: dict[str, Any], *entries: str) -> Optional[datetime]:
    epochs: list[float] = []
    keys = entries or tuple(market_data)
    for key in keys:
        value = market_data.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, dict) or item.get("date") is None:
                continue
            try:
                epochs.append(float(item["date"]))
            except (TypeError, ValueError):
                pass
    if not epochs:
        return None
    return datetime.fromtimestamp(max(epochs) / 1000.0, tz=timezone.utc)


def _parse_market_data_quote(
    symbol: str,
    market_data: dict[str, Any],
    previous: Optional[Quote] = None,
) -> Quote:
    received_at = datetime.now(timezone.utc)
    bid = _entry_number(market_data.get("BI"), "price") if "BI" in market_data else (previous.bid if previous else 0.0)
    ask = _entry_number(market_data.get("OF"), "price") if "OF" in market_data else (previous.ask if previous else 0.0)
    bid_size = _entry_number(market_data.get("BI"), "size") if "BI" in market_data else (previous.bid_size if previous else 0.0)
    ask_size = _entry_number(market_data.get("OF"), "size") if "OF" in market_data else (previous.ask_size if previous else 0.0)
    last = _entry_number(market_data.get("LA"), "price") if "LA" in market_data else (previous.last if previous else 0.0)
    if "NV" in market_data:
        volume = _entry_number(market_data.get("NV"), "size", "price")
    elif "TV" in market_data:
        volume = _entry_number(market_data.get("TV"), "size", "price")
    else:
        volume = previous.volume if previous else 0.0

    if last <= 0:
        last = (
            _entry_number(market_data.get("ACP"), "price")
            or _entry_number(market_data.get("CL"), "price")
        )

    return Quote(
        symbol=symbol,
        bid=bid,
        ask=ask,
        last=last,
        volume=volume,
        timestamp=received_at,
        bid_size=bid_size,
        ask_size=ask_size,
        source="PRIMARY_WS",
        received_at=received_at,
        market_ts=_market_timestamp(market_data) or (previous.market_ts if previous else None),
        book_ts=(
            _market_timestamp(market_data, "BI", "OF")
            or (previous.book_ts if previous else received_at)
        ),
    )


class PrimaryProvider(MarketDataProvider):
    """GGAL option chain streamed from Primary Trading API."""

    _UNDERLYING = "GGAL"
    _DEFAULT_MARKET_ID = "ROFX"

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        ws_url: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        spot_symbol: Optional[str] = None,
        caucion_symbol: Optional[str] = None,
        connect_timeout_s: float = 10.0,
        websocket_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._base_url = (base_url or settings.primary_base_url).rstrip("/")
        self._ws_url = (ws_url or settings.primary_ws_url).rstrip("/")
        self._user = user or settings.primary_user
        self._password = password or settings.primary_password
        self._configured_spot_symbol = spot_symbol or settings.primary_spot_symbol
        self._caucion_symbol = caucion_symbol or settings.primary_caucion_symbol
        self._connect_timeout_s = connect_timeout_s
        self._websocket_factory = websocket_factory

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "optionsdesk/0.1"})
        self._token = ""
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._open_event = threading.Event()
        self._ready_event = threading.Event()
        self._ws_app: Any = None
        self._ws_thread: Optional[threading.Thread] = None
        self._connected = False

        self._spot_symbol = ""
        self._option_symbols: dict[str, str] = {}
        self._market_ids: dict[str, str] = {}
        self._expiry_calendar: dict[str, date] = {}
        self._min_price_increments: dict[str, float] = {}
        self._spot: Optional[Quote] = None
        self._caucion: Optional[Quote] = None
        self._options: dict[str, Quote] = {}
        self._connected_monotonic = 0.0
        self._last_update_monotonic = 0.0
        self._last_reconnect_attempt = 0.0
        self._token_date: Optional[date] = None
        self._discovery_date: Optional[date] = None

        self._req_count = 0
        self._timeout_count = 0
        self._retry_count = 0
        self._last_latency_ms: Optional[float] = None
        self._last_success_ts: Optional[datetime] = None
        self._last_error = ""

    def connect(self) -> None:
        if not (self._base_url and self._user and self._password):
            raise RuntimeError(
                "Faltan PRIMARY_BASE_URL, PRIMARY_USER y PRIMARY_PASSWORD en .env"
            )
        if not self._ws_url:
            self._ws_url = _derive_ws_url(self._base_url)

        with self._connect_lock:
            self._close_websocket()
            self._clear_quotes()
            self._ensure_authenticated()
            self._ensure_discovered()
            self._connect_websocket()
        logger.info(
            "PrimaryProvider conectado: spot=%s, opciones=%d",
            self._spot_symbol,
            len(self._option_symbols),
        )

    def disconnect(self) -> None:
        with self._connect_lock:
            self._close_websocket()
        logger.info("PrimaryProvider desconectado")

    def reconnect(self) -> None:
        logger.warning("Reconectando Primary WebSocket (datos stale)...")
        with self._connect_lock:
            self._close_websocket()
            self._clear_quotes()
            self._ensure_authenticated()
            self._ensure_discovered()
            self._connect_websocket()

    def is_connected(self) -> bool:
        return self._connected

    def is_stale(self, threshold_s: int = 60) -> bool:
        if not self._connected:
            return True
        if self._last_update_monotonic == 0.0:
            return (
                self._connected_monotonic > 0.0
                and (time.monotonic() - self._connected_monotonic) > threshold_s
            )
        return (time.monotonic() - self._last_update_monotonic) > threshold_s

    def ensure_realtime_connection(self) -> None:
        """Recover a dropped/stale WS during the regular market session."""
        if not _is_byma_market_session() or not self.is_stale():
            return
        now = time.monotonic()
        if now - self._last_reconnect_attempt < _RECONNECT_COOLDOWN_S:
            return
        self._last_reconnect_attempt = now
        try:
            self.reconnect()
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Primary reconexion fallida: %s", exc)

    def wait_until_ready(self, timeout_s: float = _CHAIN_WAIT_S) -> bool:
        return self._ready_event.wait(timeout_s)

    def get_health(self) -> MarketDataHealth:
        with self._lock:
            options = list(self._options.values())
            spot = self._spot
        ws_age_s = (
            max(time.monotonic() - self._last_update_monotonic, 0.0)
            if self._last_update_monotonic > 0
            else None
        )
        warmup_complete = (
            self._connected_monotonic > 0
            and time.monotonic() - self._connected_monotonic >= settings.primary_warmup_s
        )
        spot_fresh = spot is not None and not spot.is_stale(settings.scalping_spot_max_age_s)
        socket_fresh = ws_age_s is not None and ws_age_s <= settings.scalping_ws_max_age_s
        ticket_capable = (
            self.is_connected()
            and warmup_complete
            and spot_fresh
            and socket_fresh
            and settings.primary_shadow_complete
        )
        blocking_reason = ""
        if not self.is_connected():
            blocking_reason = "Primary WebSocket desconectado"
        elif not warmup_complete:
            blocking_reason = "Primary WebSocket calentando"
        elif not socket_fresh or not spot_fresh:
            blocking_reason = "Primary WebSocket sin datos frescos"
        elif not settings.primary_shadow_complete:
            blocking_reason = "Rueda shadow Primary pendiente"
        coverage = (
            len(options) / len(self._option_symbols) * 100.0
            if self._option_symbols
            else 0.0
        )
        return MarketDataHealth(
            source="PRIMARY",
            connected=self.is_connected(),
            last_success_ts=self._last_success_ts,
            last_error=self._last_error,
            last_latency_ms=self._last_latency_ms,
            requests=self._req_count,
            timeouts=self._timeout_count,
            retries=self._retry_count,
            options_seen=len(self._option_symbols),
            options_tradeable=sum(1 for q in options if q.bid > 0 and q.ask > 0),
            ticket_capable=ticket_capable,
            ws_age_s=round(ws_age_s, 2) if ws_age_s is not None else None,
            warmup_complete=warmup_complete,
            coverage_pct=round(coverage, 2),
            blocking_reason=blocking_reason,
            operational_mode=(
                "TICKET_READY"
                if ticket_capable
                else "PRIMARY_WARMING"
                if self.is_connected() and not warmup_complete
                else "SHADOW"
                if self.is_connected()
                else "BLOCKED"
            ),
        )

    def get_spot(self) -> Optional[Quote]:
        self.ensure_realtime_connection()
        with self._lock:
            return self._spot

    def get_options_chain(self) -> Optional[OptionsChain]:
        self.ensure_realtime_connection()
        if not self._ready_event.is_set():
            self.wait_until_ready()
        with self._lock:
            spot = self._spot
            options = dict(self._options)
            expiry_calendar = dict(self._expiry_calendar)
            min_price_increments = dict(self._min_price_increments)
        if spot is None:
            return None
        return OptionsChain(
            underlying=self._UNDERLYING,
            spot=spot,
            options=options,
            timestamp=datetime.now(timezone.utc),
            expiry_calendar=expiry_calendar,
            min_price_increments=min_price_increments,
        )

    def get_caucion_tna(self, days: int = 30) -> Optional[float]:  # noqa: ARG002
        with self._lock:
            caucion = self._caucion
        if caucion is not None:
            value = caucion.last or caucion.mid
            if value > 0:
                return value
        return settings.default_caucion_tna

    def get_account_snapshot(self, account_name: str = "") -> AccountSnapshot:
        """Read positions through Primary Risk API. This method never routes orders."""
        account = account_name or settings.primary_account_name
        snapshot = AccountSnapshot(
            account_name=account,
            source="PRIMARY_RISK",
            fetched_at=datetime.now(timezone.utc),
        )
        if not account:
            snapshot.error = "Falta PRIMARY_ACCOUNT_NAME"
            return snapshot
        try:
            response = self._session.get(
                f"{self._base_url}/rest/position/getPositions/{urlquote(account)}",
                auth=(self._user, self._password),
                timeout=10,
            )
            if response.status_code >= 400:
                snapshot.error = f"Primary Risk HTTP {response.status_code}"
                return snapshot
            payload = response.json()
            if not isinstance(payload, list):
                snapshot.error = "Primary Risk devolvio un formato inesperado"
                return snapshot
            snapshot.positions = [item for item in payload if isinstance(item, dict)]
            snapshot.available = True
            return snapshot
        except requests.RequestException as exc:
            snapshot.error = str(exc)
            return snapshot

    def _authenticate(self) -> None:
        response = self._session.post(
            f"{self._base_url}/{_AUTH_PATH}",
            headers={"X-Username": self._user, "X-Password": self._password},
            timeout=10,
        )
        if not response.ok:
            raise RuntimeError(f"Primary auth fallo: HTTP {response.status_code}")
        token = response.headers.get("X-Auth-Token", "")
        if not token:
            raise RuntimeError("Primary auth no devolvio X-Auth-Token")
        self._token = token
        self._token_date = _today_ba()

    def _ensure_authenticated(self) -> None:
        if not self._token or self._token_date != _today_ba():
            self._authenticate()

    def _get_json(self, path: str) -> Optional[dict[str, Any]]:
        for attempt in range(_RETRIES + 1):
            self._req_count += 1
            t0 = time.perf_counter()
            try:
                response = self._session.get(
                    f"{self._base_url}/{path}",
                    headers={"X-Auth-Token": self._token},
                    timeout=10,
                )
                self._last_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                if response.status_code == 401 and attempt < _RETRIES:
                    self._retry_count += 1
                    self._authenticate()
                    continue
                if response.status_code in (429, 500, 502, 503, 504) and attempt < _RETRIES:
                    self._retry_count += 1
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    self._last_error = f"HTTP {response.status_code}"
                    logger.warning("Primary GET %s fallo: %s", path, self._last_error)
                    return None
                self._last_success_ts = datetime.now(timezone.utc)
                self._last_error = ""
                payload = response.json()
                return payload if isinstance(payload, dict) else None
            except requests.Timeout as exc:
                self._timeout_count += 1
                self._last_error = f"timeout: {exc}"
            except requests.RequestException as exc:
                self._last_error = str(exc)

            if attempt < _RETRIES:
                self._retry_count += 1
                time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
        logger.warning("Primary GET %s fallo: %s", path, self._last_error)
        return None

    def _discover_instruments(self) -> None:
        payload = self._get_json(_DETAILS_PATH) or self._get_json(_ALL_INSTRUMENTS_PATH)
        items = payload.get("instruments", []) if payload else []
        if not isinstance(items, list) or not items:
            raise RuntimeError("Primary no devolvio instrumentos")
        self._configure_instruments(items)
        self._discovery_date = _today_ba()

    def _ensure_discovered(self) -> None:
        if not self._option_symbols or self._discovery_date != _today_ba():
            self._discover_instruments()

    def _configure_instruments(self, items: list[dict[str, Any]]) -> None:
        option_symbols: dict[str, str] = {}
        market_ids: dict[str, str] = {}
        expiry_calendar: dict[str, date] = {}
        min_price_increments: dict[str, float] = {}
        available_symbols: list[str] = []

        for item in items:
            instrument_id = item.get("instrumentId") or {}
            symbol = str(instrument_id.get("symbol") or item.get("symbol") or "").strip()
            if not symbol:
                continue
            market_ids[symbol] = str(
                instrument_id.get("marketId")
                or item.get("marketId")
                or self._DEFAULT_MARKET_ID
            ).strip()
            available_symbols.append(symbol)
            try:
                increment = float(item.get("minPriceIncrement") or 0.0)
            except (TypeError, ValueError):
                increment = 0.0
            if increment > 0:
                min_price_increments[symbol] = increment
            canonical = _canonical_option_symbol(symbol)
            if canonical is None:
                continue
            maturity = _parse_maturity(item.get("maturityDate"))
            if maturity is not None and maturity < _today_ba():
                continue
            option_symbols[symbol] = canonical
            code_match = _EXPIRY_CODE_RE.search(canonical)
            if maturity is not None and code_match is not None:
                expiry_calendar[code_match.group(1)] = maturity

        spot_symbol = self._configured_spot_symbol or self._select_spot_symbol(available_symbols)
        if not spot_symbol:
            raise RuntimeError(
                "Primary no encontro el spot GGAL. Configura PRIMARY_SPOT_SYMBOL."
            )
        if not option_symbols:
            raise RuntimeError("Primary no encontro opciones GGAL (GFGC/GFGV)")

        with self._lock:
            self._spot_symbol = spot_symbol
            self._option_symbols = option_symbols
            self._market_ids = market_ids
            self._expiry_calendar = expiry_calendar
            self._min_price_increments = {
                option_symbols.get(symbol, symbol): increment
                for symbol, increment in min_price_increments.items()
            }
            self._options = {
                symbol: quote
                for symbol, quote in self._options.items()
                if symbol in option_symbols.values()
            }

    def _select_spot_symbol(self, symbols: list[str]) -> str:
        matches = [
            symbol for symbol in symbols
            if _GGAL_TOKEN_RE.search(symbol) and _canonical_option_symbol(symbol) is None
        ]
        return min(matches, key=self._spot_symbol_rank) if matches else ""

    @staticmethod
    def _spot_symbol_rank(symbol: str) -> tuple[int, int]:
        upper = symbol.upper()
        if upper == "MERV - XMEV - GGAL - 24HS":
            return (0, len(symbol))
        if upper.endswith(" - 24HS"):
            return (1, len(symbol))
        if upper == "GGAL":
            return (2, len(symbol))
        if upper.endswith(" - CI"):
            return (3, len(symbol))
        if upper.endswith(" - 48HS"):
            return (4, len(symbol))
        return (5, len(symbol))

    def _connect_websocket(self) -> None:
        if self._websocket_factory is None:
            try:
                import websocket
            except ImportError as exc:
                raise RuntimeError(
                    "websocket-client no instalado. Ejecuta: pip install websocket-client"
                ) from exc
            self._websocket_factory = websocket.WebSocketApp

        self._open_event.clear()
        self._ws_app = self._websocket_factory(
            self._ws_url,
            header=[f"X-Auth-Token: {self._token}"],
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        self._ws_thread = threading.Thread(
            target=self._run_websocket,
            name="primary-market-data",
            daemon=True,
        )
        self._ws_thread.start()
        if not self._open_event.wait(self._connect_timeout_s) or not self._connected:
            self._close_websocket()
            raise RuntimeError("Primary WebSocket no pudo conectarse")

    def _run_websocket(self) -> None:
        try:
            self._ws_app.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as exc:
            self._on_ws_error(self._ws_app, exc)

    def _close_websocket(self) -> None:
        ws_app = self._ws_app
        self._connected = False
        self._connected_monotonic = 0.0
        self._open_event.clear()
        self._ready_event.clear()
        if ws_app is not None:
            try:
                ws_app.close()
            except Exception:
                pass
        thread = self._ws_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2)
        self._ws_app = None
        self._ws_thread = None

    def _clear_quotes(self) -> None:
        with self._lock:
            self._spot = None
            self._caucion = None
            self._options = {}
        self._ready_event.clear()
        self._last_update_monotonic = 0.0

    def _on_ws_open(self, ws: Any) -> None:
        try:
            self._connected = True
            self._connected_monotonic = time.monotonic()
            self._send_subscriptions(ws)
            self._last_error = ""
        except Exception as exc:
            self._connected = False
            self._last_error = str(exc)
            logger.warning("Primary no pudo suscribirse: %s", exc)
        finally:
            self._open_event.set()

    def _send_subscriptions(self, ws: Any) -> None:
        symbols = [self._spot_symbol, *self._option_symbols]
        if self._caucion_symbol:
            symbols.append(self._caucion_symbol)
        symbols = list(dict.fromkeys(symbol for symbol in symbols if symbol))

        for offset in range(0, len(symbols), _SUBSCRIPTION_BATCH_SIZE):
            batch = symbols[offset:offset + _SUBSCRIPTION_BATCH_SIZE]
            ws.send(json.dumps({
                "type": "smd",
                "level": settings.primary_ws_level,
                "depth": settings.primary_ws_depth,
                "entries": list(_WS_ENTRIES),
                "products": [
                    {
                        "symbol": symbol,
                        "marketId": self._market_ids.get(
                            symbol, self._DEFAULT_MARKET_ID
                        ),
                    }
                    for symbol in batch
                ],
            }))

    def _on_ws_message(self, ws: Any, raw_message: str) -> None:  # noqa: ARG002
        try:
            message = json.loads(raw_message)
            if message.get("status") == "ERROR":
                self._last_error = str(message)
                logger.warning("Primary WebSocket error: %s", message)
                return
            if str(message.get("type", "")).upper() != "MD":
                return

            instrument_id = message.get("instrumentId") or {}
            raw_symbol = str(instrument_id.get("symbol") or "").strip()
            market_data = message.get("marketData") or {}
            if not raw_symbol or not isinstance(market_data, dict):
                return

            with self._lock:
                if raw_symbol == self._spot_symbol:
                    self._spot = _parse_market_data_quote(
                        self._UNDERLYING, market_data, self._spot
                    )
                elif raw_symbol == self._caucion_symbol:
                    self._caucion = _parse_market_data_quote(
                        raw_symbol, market_data, self._caucion
                    )
                else:
                    canonical = self._option_symbols.get(raw_symbol)
                    if canonical is None:
                        return
                    self._options[canonical] = _parse_market_data_quote(
                        canonical, market_data, self._options.get(canonical)
                    )
                ready = self._spot is not None and bool(self._options)
            if ready:
                self._ready_event.set()
            self._last_update_monotonic = time.monotonic()
            self._last_success_ts = datetime.now(timezone.utc)
            self._last_latency_ms = _market_data_latency_ms(market_data)
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Primary mensaje WebSocket invalido: %s", exc)

    def _on_ws_error(self, ws: Any, error: Any) -> None:  # noqa: ARG002
        self._connected = False
        self._last_error = str(error)
        logger.warning("Primary WebSocket fallo: %s", error)

    def _on_ws_close(
        self,
        ws: Any,  # noqa: ARG002
        close_status_code: Optional[int] = None,
        close_msg: Optional[str] = None,
    ) -> None:
        self._connected = False
        if close_status_code is not None:
            self._last_error = f"WebSocket cerrado: {close_status_code} {close_msg or ''}".strip()
