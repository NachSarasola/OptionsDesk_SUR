"""Proveedor de datos via API REST de InvertirOnline (IOL)."""
from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
import time
from datetime import date, datetime
from typing import Optional

import requests

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import (
    MarketDataHealth,
    MarketDataProvider,
    OptionsChain,
    Quote,
)

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://api.invertironline.com/token"
_BASE_URL = "https://api.invertironline.com/api/v2"
_TOKEN_BUFFER_S = 60
_REFRESH_TTL_DAYS = 7
_GET_RETRIES = 2
_RETRY_BASE_SLEEP_S = 0.30


class _TokenManager:
    """Gestiona access_token + refresh_token con renovacion thread-safe."""

    def __init__(self, user: str, password: str) -> None:
        self._user = user
        self._password = password
        self._lock = threading.Lock()
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0.0
        self._refresh_expires_at: float = 0.0

    def get_headers(self) -> dict[str, str]:
        token = self._ensure_token()
        return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    def _ensure_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._access_token and now < self._expires_at - _TOKEN_BUFFER_S:
                return self._access_token
            if self._refresh_token and now < self._refresh_expires_at - _TOKEN_BUFFER_S:
                self._do_refresh()
            else:
                self._do_login()
            if not self._access_token:
                raise RuntimeError("IOL: no se pudo obtener access_token")
            return self._access_token

    def _do_login(self) -> None:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "username": self._user,
                "password": self._password,
                "grant_type": "password",
            },
            timeout=10,
        )
        resp.raise_for_status()
        self._parse_token_response(resp.json())
        logger.info("IOL: token obtenido via login")

    def _do_refresh(self) -> None:
        resp = requests.post(
            _TOKEN_URL,
            data={
                "refresh_token": self._refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=10,
        )
        if not resp.ok:
            logger.warning("IOL: refresh_token fallo (%d), re-login", resp.status_code)
            self._do_login()
            return
        self._parse_token_response(resp.json())

    def _parse_token_response(self, data: dict) -> None:
        self._access_token = data["access_token"]
        self._refresh_token = data.get("refresh_token")
        expires_in = float(data.get("expires_in", 900))
        self._expires_at = time.time() + expires_in

        refresh_exp_str = data.get(".refreshexpires")
        if refresh_exp_str:
            try:
                dt = datetime.fromisoformat(str(refresh_exp_str).replace("Z", "+00:00"))
                self._refresh_expires_at = dt.timestamp()
            except Exception:
                self._refresh_expires_at = time.time() + _REFRESH_TTL_DAYS * 86400
        else:
            self._refresh_expires_at = time.time() + _REFRESH_TTL_DAYS * 86400


class IOLProvider(MarketDataProvider):
    """Datos de mercado de GGAL via API REST de IOL."""

    _MARKET = "bCBA"
    _UNDERLYING = "GGAL"

    def __init__(self) -> None:
        self._token_mgr: Optional[_TokenManager] = None
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self._session.mount("https://", adapter)
        self._connected = False

        self._req_count = 0
        self._timeout_count = 0
        self._retry_count = 0
        self._last_latency_ms: Optional[float] = None
        self._last_success_ts: Optional[datetime] = None
        self._last_error = ""
        self._options_seen = 0
        self._options_tradeable = 0
        self._options_listing: list[dict] = []
        self._options_listing_monotonic = 0.0

    def connect(self) -> None:
        if not settings.iol_user or not settings.iol_password:
            raise RuntimeError("Faltan IOL_USER e IOL_PASSWORD en .env")
        self._token_mgr = _TokenManager(settings.iol_user, settings.iol_password)
        self._token_mgr._ensure_token()
        self._connected = True
        logger.info("IOLProvider conectado")

    def disconnect(self) -> None:
        self._connected = False
        self._token_mgr = None
        logger.info("IOLProvider desconectado")

    def is_connected(self) -> bool:
        return self._connected and self._token_mgr is not None

    def get_token_manager(self) -> _TokenManager:
        """Devuelve el gestor de token vivo para reusar la sesion (p.ej. operar)."""
        if not self._token_mgr:
            raise RuntimeError("IOLProvider no conectado. Llama connect() primero.")
        return self._token_mgr

    def get_health(self) -> MarketDataHealth:
        connected = self.is_connected()
        coverage = (
            self._options_tradeable / self._options_seen * 100.0
            if self._options_seen > 0
            else 0.0
        )
        provisional = connected and settings.iol_provisional_tickets_enabled
        return MarketDataHealth(
            source="IOL",
            connected=connected,
            last_success_ts=self._last_success_ts,
            last_error=self._last_error,
            last_latency_ms=self._last_latency_ms,
            requests=self._req_count,
            timeouts=self._timeout_count,
            retries=self._retry_count,
            options_seen=self._options_seen,
            options_tradeable=self._options_tradeable,
            ticket_capable=provisional,
            warmup_complete=connected,
            coverage_pct=round(coverage, 2),
            blocking_reason="" if provisional else "IOL provisional deshabilitado",
            operational_mode="IOL_PROVISIONAL" if provisional else "RADAR_IOL",
        )

    def get_spot(self) -> Optional[Quote]:
        return self.get_quote(self._UNDERLYING)

    def get_quote(self, symbol: str) -> Optional[Quote]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return None
        data = self._get(f"{self._MARKET}/Titulos/{symbol}/Cotizacion")
        if not isinstance(data, dict):
            return None
        return _parse_quote(symbol, data, source="IOL_REST")

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        clean = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))
        if not clean:
            return {}
        quotes: dict[str, Quote] = {}
        max_workers = min(8, len(clean))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.get_quote, symbol): symbol for symbol in clean}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    quote = future.result()
                except Exception as exc:
                    logger.debug("IOL: get_quote fallo para %s: %s", symbol, exc)
                    continue
                if quote is not None:
                    quotes[quote.symbol.upper()] = quote
        return quotes

    def get_options_chain(self) -> Optional[OptionsChain]:
        spot = self.get_spot()
        if spot is None:
            self._options_seen = 0
            self._options_tradeable = 0
            return None
        spot_px = spot.mid if spot.mid > 0 else spot.last
        if spot_px <= 0:
            self._last_error = "spot GGAL invalido"
            self._options_seen = 0
            self._options_tradeable = 0
            return None

        data = self._get_options_listing()
        if not isinstance(data, list):
            self._options_seen = 0
            self._options_tradeable = 0
            return None

        self._options_seen = len(data)
        selected_items: list[tuple[float, str, dict]] = []
        observed_expiry_calendar: dict[str, date] = {}

        from optionsdesk.core.instruments import parse_option_symbol

        dummy_cal = {chr(i): date.today() for i in range(ord("A"), ord("Z") + 1)}
        for item in data:
            sym = str(item.get("simbolo", "")).upper().strip()
            if not (sym and sym.startswith("GFG")):
                continue
            match = re.match(r"^GFG[CV]\d+(?:[.,]\d+)?([A-Z]+)$", sym)
            expiry_raw = str(item.get("fechaVencimiento", "") or "")[:10]
            if match and expiry_raw:
                try:
                    code = match.group(1)
                    expiration = date.fromisoformat(expiry_raw)
                    previous = observed_expiry_calendar.get(code)
                    if previous is not None and previous != expiration:
                        logger.warning("IOL: codigo %s con vencimientos inconsistentes: %s / %s", code, previous, expiration)
                    observed_expiry_calendar[code] = expiration
                except ValueError:
                    logger.debug("IOL: vencimiento invalido para %s: %s", sym, expiry_raw)
            contract = parse_option_symbol(sym, {**dummy_cal, **observed_expiry_calendar})
            if not contract:
                continue

            strike = contract.strike
            while strike > spot_px * 3:
                strike /= 10.0
            while strike < spot_px / 3:
                strike *= 10.0

            if spot_px * 0.5 <= strike <= spot_px * 1.5:
                selected_items.append((abs(strike - spot_px), sym, item))

        opts: dict[str, Quote] = {}
        for _, sym, item in selected_items:
            # La cadena ya incluye la cotizacion completa de cada opcion.
            # Cuando IOL la completa, evita una llamada REST adicional.
            q = _parse_quote(sym, item, source="IOL_LISTING")
            if self._is_option_quote_usable(q, spot):
                opts[q.symbol] = q

        # En la practica, el listado agregado de IOL puede venir sin puntas aun
        # durante la rueda. Recupera solo un radar ATM acotado para mantener el
        # consumo mensual controlado hasta disponer de Primary WebSocket.
        limit = max(settings.iol_option_quotes_per_cycle, 0)
        nearest = sorted(selected_items, key=lambda row: (row[0], row[1]))[:limit]
        if nearest:
            max_workers = min(8, len(nearest))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._get_option_quote, sym): sym
                    for _, sym, _ in nearest
                }
                for future in concurrent.futures.as_completed(futures):
                    q = future.result()
                    if self._is_option_quote_usable(q, spot):
                        opts[q.symbol] = q

        self._options_tradeable = sum(1 for q in opts.values() if q.bid > 0 and q.ask > 0)
        return OptionsChain(
            underlying=self._UNDERLYING,
            spot=spot,
            options=opts,
            expiry_calendar=observed_expiry_calendar,
        )

    def _get_options_listing(self) -> Optional[list[dict]]:
        ttl_s = max(settings.iol_options_list_ttl_s, 0)
        if (
            self._options_listing
            and time.monotonic() - self._options_listing_monotonic < ttl_s
        ):
            return self._options_listing
        data = self._get(f"{self._MARKET}/Titulos/{self._UNDERLYING}/Opciones")
        if isinstance(data, list):
            self._options_listing = data
            self._options_listing_monotonic = time.monotonic()
            return data
        return None

    def _get_option_quote(self, symbol: str) -> Optional[Quote]:
        data = self._get(f"{self._MARKET}/Titulos/{symbol}/Cotizacion")
        return _parse_quote(symbol, data, source="IOL_REST") if isinstance(data, dict) else None

    @staticmethod
    def _is_option_quote_usable(q: Optional[Quote], spot: Quote) -> bool:
        if q is None:
            return False
        if q.volume > 50_000_000 and abs(q.last - spot.last) < spot.last * 0.05:
            return False
        return not (q.last == 0 and q.bid == 0 and q.ask == 0)

    def get_caucion_tna(self, days: int = 30) -> Optional[float]:
        for panel in ("general", "todos"):
            data = self._get(
                f"Cotizaciones/cauciones/{panel}/argentina",
                quiet_statuses={400, 403, 404, 500},
            )
            if isinstance(data, list) and data:
                tna = _best_caucion_tna(data, days)
                if tna is not None:
                    return tna
        return settings.default_caucion_tna

    def _get(
        self,
        path: str,
        quiet_statuses: Optional[set[int]] = None,
    ) -> Optional[dict | list]:
        if not self._token_mgr:
            raise RuntimeError("IOLProvider no conectado. Llama connect() primero.")
        quiet = quiet_statuses or set()
        url = f"{_BASE_URL}/{path}"

        for attempt in range(_GET_RETRIES + 1):
            t0 = time.perf_counter()
            self._req_count += 1
            try:
                headers = self._token_mgr.get_headers()
                resp = self._session.get(url, headers=headers, timeout=10)
                self._last_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if resp.status_code == 401:
                    logger.warning("IOL: 401 en %s, re-login", path)
                    self._token_mgr._do_login()
                    headers = self._token_mgr.get_headers()
                    resp = self._session.get(url, headers=headers, timeout=10)

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < _GET_RETRIES:
                    self._retry_count += 1
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue

                if resp.status_code >= 400:
                    self._last_error = f"HTTP {resp.status_code}"
                    if resp.status_code not in quiet:
                        logger.warning("IOL GET %s fallo: HTTP %d", path, resp.status_code)
                    return None

                self._last_success_ts = datetime.now()
                self._last_error = ""
                return resp.json()

            except requests.Timeout as exc:
                self._timeout_count += 1
                self._last_error = f"timeout: {exc}"
                if attempt < _GET_RETRIES:
                    self._retry_count += 1
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue
                logger.warning("IOL GET %s timeout: %s", path, exc)
                return None
            except requests.RequestException as exc:
                self._last_error = str(exc)
                if attempt < _GET_RETRIES:
                    self._retry_count += 1
                    time.sleep(_RETRY_BASE_SLEEP_S * (attempt + 1))
                    continue
                logger.warning("IOL GET %s fallo: %s", path, exc)
                return None

def _parse_datetime(value) -> Optional[datetime]:
    if not value or str(value).startswith("0001-01-01"):
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _parse_quote(symbol: str, data: dict, *, source: str = "IOL_REST") -> Optional[Quote]:
    """Construye un Quote desde la respuesta JSON de IOL."""
    try:
        qdata = data.get("cotizacion") if isinstance(data.get("cotizacion"), dict) else data

        bid = ask = bid_size = ask_size = 0.0
        puntas = qdata.get("puntas") or []
        if puntas:
            best = puntas[0]
            bid = float(best.get("precioCompra") or 0)
            ask = float(best.get("precioVenta") or 0)
            bid_size = float(best.get("cantidadCompra") or 0)
            ask_size = float(best.get("cantidadVenta") or 0)

        ultimo = qdata.get("ultimo")
        cierre_anterior = float(qdata.get("cierreAnterior") or 0)
        market_ts = None
        if isinstance(ultimo, dict):
            last = float(ultimo.get("precio") or 0)
            market_ts = _parse_datetime(ultimo.get("fecha"))
        else:
            ops = int(qdata.get("cantidadOperaciones") or 0)
            last_candidate = float(
                qdata.get("ultimoPrecio")
                or (ultimo if isinstance(ultimo, (int, float)) else 0)
                or 0
            )
            last = last_candidate if (ops > 0 or last_candidate > 0) else 0.0
            market_ts = _parse_datetime(qdata.get("fechaHora"))

        if symbol.upper() == "GGAL" and last <= 0:
            last = float(qdata.get("ultimoPrecio") or cierre_anterior or 0)

        # Evita tomar cierre anterior del spot para opciones sin negocio.
        if last <= 0 and cierre_anterior < 500:
            last = cierre_anterior

        received_at = datetime.now().astimezone()
        book_ts = _parse_datetime(qdata.get("fechaHora")) or received_at

        volume = float(qdata.get("volumenNominal") or qdata.get("volumeNominal") or 0)

        return Quote(
            symbol=symbol,
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            timestamp=received_at,
            bid_size=bid_size,
            ask_size=ask_size,
            source=source,
            received_at=received_at,
            market_ts=market_ts,
            book_ts=book_ts,
        )
    except Exception as exc:
        logger.debug("IOL: no se pudo parsear quote para %s: %s", symbol, exc)
        return None


_CAUCION_DAYS_RE = re.compile(r"(\d+)D", re.IGNORECASE)


def _best_caucion_tna(items: list[dict], target_days: int) -> Optional[float]:
    best_tna: Optional[float] = None
    best_diff = float("inf")
    for item in items:
        sym = str(item.get("simbolo", "")).upper()
        m = _CAUCION_DAYS_RE.search(sym)
        if not m:
            continue
        sym_days = int(m.group(1))
        tna = _extract_caucion_tna(item)
        if tna is None:
            continue
        diff = abs(sym_days - target_days)
        if diff < best_diff:
            best_tna = tna
            best_diff = diff
    return best_tna


def _extract_caucion_tna(item: dict) -> Optional[float]:
    for key in ("tasaAnual", "tna", "ultimo", "precioUltimo", "cierreAnterior"):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("precio") or val.get("tasa")
        try:
            num = float(val)
            if 0.0 < num < 2000.0:
                return num
        except (TypeError, ValueError):
            continue
    return None
