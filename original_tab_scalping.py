def _tab_scalping(
    chain: OptionsChain,
    spot: float,
    provider: MarketDataProvider,
    context,
    expiry_calendar: dict,
    spot_history_df: Optional[pd.DataFrame],
    capital: Optional[float],
    risk_profile: str,
    caucion_tna: float,
    ltf_df: Optional[pd.DataFrame] = None,
) -> None:
    from optionsdesk.signals.scalping import build_scalp_verdict, scan_all

    st.subheader("Scalping & Momentum")
    st.caption("Mesa intradia con ejecucion manual. Opera en tu broker con orden limite y registra el fill confirmado.")

    # ── Controles estables (fuera del fragment, no parpadean) ─────────────────
    ctl_cap, ctl_iv, ctl_refresh = st.columns([3, 1, 1])
    with ctl_cap:
        capital_default = int(st.session_state.get("scalp_capital", capital or settings.default_capital or 0) or 0)
        capital_input = st.number_input(
            "Capital máximo para scalping (ARS)",
            min_value=0, max_value=100_000_000, value=capital_default, step=50_000,
            help="Define sizing, riesgo por trade y bloqueo de señales.",
            key="capital_scalping_main",
        )
        capital_live = float(capital_input) if capital_input > 0 else None
        st.session_state["scalp_capital"] = float(capital_input or 0)
    with ctl_iv:
        interval_opts = {"15s": 15, "30s": 30, "1min": 60, "2min": 120, "Pausa": None}
        interval_label = st.selectbox(
            "Auto-refresh", list(interval_opts.keys()), index=1,
            help="Solo actualiza el panel de señales y el gráfico — el resto de la app no se toca.",
            key="scalp_refresh_interval",
        )
        interval_s = interval_opts[interval_label]
    with ctl_refresh:
        st.write("")
        if st.button("Forzar ahora", key="btn_refresh_scalping", use_container_width=True):
            st.session_state["force_refresh"] = True
            st.rerun()
    allow_overnight_entries = st.toggle(
        "Evaluar entrada overnight",
        value=False,
        help="Activalo solo para buscar una posicion nueva que deliberadamente se mantendra al dia siguiente.",
        key="scalp_allow_overnight_entries",
    )
    chart_timeframe = st.segmented_control(
        "Velas",
        ["Tape 1m", "Tape 5m", "Diario", "Semanal"],
        default="Tape 1m",
        key="scalp_chart_timeframe",
        help="Tape usa observaciones spot IOL desde que abriste el panel. Diario y semanal incluyen la cotizacion live actual.",
    )

    # RV diaria: se calcula una vez por carga de página (historial daily, no cambia en minutos)
    realized_vol = None
    if spot_history_df is not None and not spot_history_df.empty:
        try:
            from optionsdesk.signals.volatility import yang_zhang_volatility, realized_volatility
            realized_vol = yang_zhang_volatility(spot_history_df, window=20)
            if realized_vol is None and "close" in spot_history_df.columns:
                realized_vol = realized_volatility(spot_history_df["close"].dropna().tolist(), window=20)
        except Exception:
            realized_vol = None

    # ── Panel en vivo: se auto-actualiza sin tocar el resto de la app ─────────
    # st.fragment re-ejecuta SOLO este bloque en el intervalo indicado.
    # La cadena se consulta en cada tick: no presentamos quotes cacheadas como si fueran vivas.
    @st.fragment(run_every=interval_s)
    def _live_panel() -> None:
        # Datos frescos dentro del fragment
        try:
            from optionsdesk.signals.monitor import PositionMonitor
            positions_now = PositionMonitor(settings.open_positions_file).load_positions()
        except Exception:
            positions_now = []

        # LTF con TTL=20s — cada tick del fragment obtiene barras frescas del cache
        ltf_live = _load_ltf_history()

        try:
            chain_now: Optional[OptionsChain] = provider.get_options_chain()
        except Exception as exc:
            st.error(f"No se pudo actualizar la cadena: {exc}")
            return
        if chain_now is None:
            st.warning("Sin cadena fresca de opciones. Espera el proximo tick o revisa el feed.")
            return
        st.session_state["chain"] = chain_now
        live_expiry_calendar = _effective_expiry_calendar(chain_now, expiry_calendar)
        health = provider.get_health()
        spot_now = chain_now.spot.mid
        live_pulse = _spot_tape_pulse(spot_now, st.session_state.get("provider_key", "unknown"))
        tape_samples = list(st.session_state.get("scalp_spot_tape", []))
        tape_1m = _tape_ohlc(tape_samples, "1min")

        snap_now = None
        live_bars = (
            ltf_live if ltf_live is not None and not ltf_live.empty
            else tape_1m if len(tape_1m) >= 20
            else ltf_df
        )
        if live_bars is not None and not live_bars.empty:
            try:
                from optionsdesk.signals.technical import analyze
                snap_now = analyze(live_bars)
            except Exception:
                snap_now = None
        if snap_now is None and context is not None:
            snap_now = getattr(context, "ltf_snap", None) or getattr(context, "snap", None)

        try:
            greeks, signals = scan_all(
                chain_now, live_expiry_calendar,
                snap=snap_now, realized_vol=realized_vol, r=0.0,
                capital=capital_live, positions=positions_now, risk_profile=risk_profile,
                allow_overnight_entries=allow_overnight_entries,
                require_live_confirmation=True,
                live_confirmation=live_pulse.confirmed and health.source == "IOL" and health.connected,
                live_direction=live_pulse.direction,
            )
        except Exception as exc:
            st.error(f"Scalping scan error: {exc}")
            return

        actionables = [s for s in signals if s.confidence == "actionable"]
        watchlist   = [s for s in signals if s.confidence != "actionable"]
        tradeable   = [g for g in greeks.values() if g.is_tradeable]

        if not capital_live or capital_live <= 0:
            actionables = []

        # ── 6 cards de estado ─────────────────────────────────────────────────
        h1, h2, h3, h4, h5, h6 = st.columns(6)
        latency = f"{health.last_latency_ms:.0f} ms" if health.last_latency_ms is not None else "-"
        now_ba = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
        session_end = now_ba.replace(hour=17, minute=0, second=0, microsecond=0)
        h_val = now_ba.hour + now_ba.minute / 60
        session_is_open = now_ba.weekday() < 5 and 11 <= h_val < 17
        mins_left = max((session_end - now_ba).total_seconds() / 60, 0) if session_is_open else 0
        session_color = "#22c55e" if mins_left > 30 else ("#f59e0b" if mins_left > 10 else "#ef4444")
        session_phase = (
            getattr(snap_now, "session_phase", None) or
            ("cerrada" if (h_val < 11 or h_val >= 17)
             else ("apertura" if h_val < 11.33 else ("cierre" if h_val >= 16.75 else "regular")))
        )
        verdict = build_scalp_verdict(
            signals,
            feed_live=health.source == "IOL" and health.connected,
            capital=capital_live,
            pulse=live_pulse,
            allow_overnight_entries=allow_overnight_entries,
            session_phase=(
                "closed" if not session_is_open
                else "close" if mins_left <= settings.scalping_eod_window_min
                else "open" if h_val < 11.33
                else "regular"
            ),
        )
        h1.markdown(_scalp_card("FEED", health.source, "conectado" if health.connected else "revisar", "#22c55e" if health.connected else "#ef4444"), unsafe_allow_html=True)
        h2.markdown(_scalp_card("LATENCIA", latency, f"{health.timeouts} timeouts acumulados", "#22c55e" if not health.last_error else "#f59e0b"), unsafe_allow_html=True)
        h3.markdown(_scalp_card("CADENA", f"{len(tradeable)}/{len(greeks)}", "operable/parseada", "#22c55e" if tradeable else "#ef4444"), unsafe_allow_html=True)
        h4.markdown(_scalp_card("SESIÓN", f"{int(mins_left)}m", session_phase, session_color), unsafe_allow_html=True)
        h5.markdown(_scalp_card("RV 20D", f"{realized_vol * 100:.1f}%" if realized_vol else "-", "Yang-Zhang/close", "#71717a"), unsafe_allow_html=True)
        pulse_label = (
            "ALCISTA" if live_pulse.confirmed and live_pulse.direction == "BULL"
            else "BAJISTA" if live_pulse.confirmed and live_pulse.direction == "BEAR"
            else "FORMANDO" if live_pulse.observations < 5 or live_pulse.span_s < 60
            else "NEUTRAL"
        )
        h6.markdown(_scalp_card("PULSO LIVE", pulse_label, f"{live_pulse.observations}/5 obs | {live_pulse.span_s:.0f}/60s", "#22c55e" if live_pulse.confirmed else "#f59e0b"), unsafe_allow_html=True)

        # ── Pre-flight ────────────────────────────────────────────────────────
        preflight = []
        if not settings.is_iol_configured() or health.source != "IOL":
            preflight.append("No estás usando IOL como fuente primaria.")
        if capital_live is None or capital_live <= 0:
            preflight.append("Cargá capital para sizing real.")
        if not live_pulse.confirmed:
            preflight.append("Esperando confirmacion direccional del tape live de GGAL.")
        if snap_now is None:
            preflight.append("Sin contexto tecnico disponible.")
        if realized_vol is None:
            preflight.append("Sin RV: gamma scalp menos confiable.")
        if health.last_error:
            preflight.append(f"Ultimo error del feed: {health.last_error}.")

        if preflight:
            st.caption("Pendiente: " + " | ".join(preflight))
        else:
            st.caption("Estado: feed, contexto, RV y capital listos.")

        # ── Gráfico de velas + niveles ────────────────────────────────────────
        if verdict.signal is not None:
            best = verdict.signal
            side = "CALL" if verdict.decision == "BUY_CALL_NOW" else "PUT"
            st.success(
                f"**COMPRAR {side} AHORA: {best.symbol}** | limite {_scalp_money(best.plan_entry, 2)} | "
                f"stop {_scalp_money(best.plan_sl, 2)} | TP {_scalp_money(best.plan_tp, 2)} | "
                f"PoP {best.probability_of_profit * 100:.0f}% | EV {_scalp_money(best.expected_value_ars)}/lote"
            )
            st.caption(
                f"{verdict.mode} | {best.playbook} | {verdict.reason} | "
                f"GGAL stop {_scalp_money(best.underlying_stop, 2)} / target {_scalp_money(best.underlying_target, 2)} | "
                f"time stop {best.time_stop_min}min"
            )
        else:
            st.warning(f"**{_scalp_wait_message(verdict)}**")

        daily_live = _daily_with_live_spot(spot_history_df, tape_samples, spot_now)
        if chart_timeframe == "Tape 5m":
            chart_df = _tape_ohlc(tape_samples, "5min")
            chart_label = "Tape IOL 5m desde apertura del panel"
            chart_smas = {"SMA5": 5, "SMA9": 9}
            chart_max_bars = 72
            chart_volume = False
        elif chart_timeframe == "Diario":
            chart_df = daily_live
            chart_label = "Diario con vela live de hoy"
            chart_smas = {"SMA9": 9, "SMA20": 20}
            chart_max_bars = 90
            chart_volume = True
        elif chart_timeframe == "Semanal":
            chart_df = _weekly_from_daily(daily_live)
            chart_label = "Semanal con semana actual live"
            chart_smas = {"SMA5": 5, "SMA20": 20}
            chart_max_bars = 72
            chart_volume = True
        else:
            chart_df = tape_1m
            chart_label = "Tape IOL 1m desde apertura del panel"
            chart_smas = {"SMA5": 5, "SMA9": 9}
            chart_max_bars = 120
            chart_volume = False
        levels = [{"y": spot_now, "label": f"${spot_now:,.0f}", "color": "#9ca3af", "dash": "dot"}]

        if actionables:
            sig_labels = [
                f"{s.action} {s.symbol} · PoP {s.probability_of_profit * 100:.0f}% · R:R {s.plan_rr:.1f}"
                for s in actionables
            ]
            pick = st.selectbox(
                "Señal en el gráfico",
                range(len(actionables)),
                format_func=lambda i: sig_labels[i],
                key="scalp_chart_pick",
            )
            csig = actionables[pick]
            if csig.underlying_target > 0:
                levels.append({"y": csig.underlying_target, "label": f"TP ${csig.underlying_target:,.0f}", "color": "#26a69a", "dash": "dash"})
            if csig.underlying_stop > 0:
                levels.append({"y": csig.underlying_stop, "label": f"Stop ${csig.underlying_stop:,.0f}", "color": "#ef5350", "dash": "dash"})

        _candlestick_chart(
            chart_df, height=400,
            smas=chart_smas,
            levels=levels,
            show_volume=chart_volume,
            max_bars=chart_max_bars,
        )
        st.caption(
            f"{chart_label} | actualizacion cada {interval_label}."
        )

        # ── Modo overnight: aviso solo cuando el operador lo habilito ─────────
        overnight_active = any(
            "setup_overnight" in getattr(s, "risk_flags", []) for s in signals
        )
        if overnight_active:
            st.info(
                "Modo overnight activo: filtros mas estrictos (DTE>=5, delta>=0.35), "
                "sizing al 50% por gap risk. Los trades listados estan pensados para "
                "mantener hasta la apertura del proximo dia."
            )

        # ── Señales accionables ───────────────────────────────────────────────
        st.markdown("### Señales accionables")
        if not actionables:
            st.caption("Sin ticket habilitado: el veredicto superior indica la condicion pendiente.")
        else:
            cols = st.columns(min(3, len(actionables)))
            for i, sig in enumerate(actionables[:3]):
                color = _signal_color(sig)
                with cols[i]:
                    st.markdown(
                        f"<div style='background:rgba(26,26,36,0.86);border-top:3px solid {color};"
                        f"border-radius:8px;padding:12px;'>"
                        f"<div style='font-size:0.72rem;color:#71717A;text-transform:uppercase;'>{sig.playbook} · {sig.urgency}</div>"
                        f"<div style='font-size:1.2rem;font-weight:700;margin-top:2px;'>{sig.action} {sig.symbol}</div>"
                        f"<div style='font-size:0.78rem;color:{color};font-weight:600;margin-top:4px;'>{sig.expected_move}</div>"
                        f"<div style='font-size:0.78rem;color:#d4d4d8;margin-top:8px;line-height:1.35;'>{sig.rationale}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    with st.expander("Ticket", expanded=True):
                        valid_txt = sig.valid_until.split("T")[-1][:8] if sig.valid_until else "-"
                        st.write(f"Entrada: **{_scalp_money(sig.plan_entry, 2)}** ({sig.entry_style})")
                        st.write(f"Stop opción: **{_scalp_money(sig.plan_sl, 2)}** · TP opción: **{_scalp_money(sig.plan_tp, 2)}**")
                        st.write(f"Stop GGAL: **{_scalp_money(sig.underlying_stop, 2)}** · TP GGAL: **{_scalp_money(sig.underlying_target, 2)}**")
                        st.caption(
                            f"PoP {sig.probability_of_profit * 100:.0f}% · EV {_scalp_money(sig.expected_value_ars)} · "
                            f"R:R {sig.plan_rr:.2f} · Riesgo {_scalp_money(sig.planned_risk_ars)}/lote · "
                            f"Fricción {sig.friction_pct:.1f}% · Fill {getattr(sig, 'fill_probability', 0.0) * 100:.0f}% · "
                            f"Time stop {getattr(sig, 'time_stop_min', 20)}min · válido hasta {valid_txt}"
                        )
                        if getattr(sig, "warning_flags", []):
                            st.caption("Advertencias: " + _scalp_flags_text(sig.warning_flags))
                        if sig.suggested_lots <= 0:
                            st.error("Sizing bloqueado — cargá capital.")
                        else:
                            lotes = st.number_input(
                                "Lotes", min_value=1, max_value=max(int(sig.suggested_lots), 1),
                                value=max(int(sig.suggested_lots or 1), 1), step=1,
                                key=f"scalp_lotes_{sig.symbol}_{i}",
                            )
                            if st.button("Registrar fill manual", key=f"scalp_reg_{sig.symbol}_{i}", type="primary"):
                                _simulate_scalp_trade(sig, spot_now, lotes, caucion_tna)
                                st.success("Fill registrado en Portfolio. El dashboard no envia ordenes al broker.")

        # ── Tabla resumen ─────────────────────────────────────────────────────
        def _max_loss_label(s) -> str:
            unc = getattr(s, "max_loss_uncapped_ars", s.max_loss_ars)
            if isinstance(unc, float) and unc == float("inf"):
                return f"{_scalp_money(s.max_loss_ars)} (sin stop: ilimitado)"
            if unc > s.max_loss_ars * 1.01:
                return f"{_scalp_money(s.max_loss_ars)} (peor caso {_scalp_money(unc)})"
            return _scalp_money(s.max_loss_ars)

        def _signals_df(items: list) -> pd.DataFrame:
            rows = []
            for s in items:
                g = greeks.get(s.symbol)
                rows.append({
                    "Acción": s.action, "Símbolo": s.symbol,
                    "Setup": s.playbook or s.signal_type,
                    "Score": round(s.score, 1),
                    "PoP": f"{s.probability_of_profit * 100:.0f}%",
                    "EV/lote": _scalp_money(s.expected_value_ars),
                    "Entrada": _scalp_money(s.plan_entry, 2),
                    "Stop": _scalp_money(s.plan_sl, 2),
                    "TP": _scalp_money(s.plan_tp, 2),
                    "R:R": f"{s.plan_rr:.2f}",
                    "Riesgo/lote": _scalp_money(s.planned_risk_ars),
                    "Max loss": _max_loss_label(s),
                    "Fill": f"{getattr(s, 'fill_probability', 0.0) * 100:.0f}%",
                    "Spread": f"{s.spread_pct:.1f}%",
                    "Edad": f"{getattr(g, 'quote_age_s', 0.0):.0f}s" if g else "-",
                    "Bloqueos": _scalp_flags_text(getattr(s, "blocking_flags", [])) or "-",
                })
            return pd.DataFrame(rows)

        if actionables:
            st.dataframe(_signals_df(actionables), hide_index=True, use_container_width=True)

        # ── Watchlist ─────────────────────────────────────────────────────────
        with st.expander(f"Radar y bloqueos ({len(watchlist)})", expanded=False):
            if not watchlist:
                st.caption("Sin contratos en radar.")
            else:
                reason_counts: dict[str, int] = {}
                for s in watchlist:
                    for flag in getattr(s, "blocking_flags", []) or s.risk_flags:
                        reason_counts[flag] = reason_counts.get(flag, 0) + 1
                if reason_counts:
                    top = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    st.caption("Bloqueos: " + " | ".join(f"{_scalp_flag_label(k)} ({v})" for k, v in top))
                st.dataframe(_signals_df(watchlist[:40]), hide_index=True, use_container_width=True)

        # ── Griegas (expander para no ocupar pantalla por default) ────────────
        with st.expander("Griegas de la cadena", expanded=False):
            rows = []
            for g in sorted(greeks.values(), key=lambda x: (x.days, abs(x.moneyness_pct))):
                rows.append({
                    "Símbolo": g.symbol, "Tipo": "C" if g.option_type == "C" else "P",
                    "Strike": g.strike, "DTE": g.days,
                    "Bid": g.bid, "Ask": g.ask, "Spread%": round(g.spread_pct, 1),
                    "IV%": f"{g.iv * 100:.1f}" if g.iv else "-",
                    "Delta": round(g.delta, 3), "Gamma": round(g.gamma, 5),
                    "Theta": round(g.theta, 2), "Vol": g.volume,
                    "Edad": f"{g.quote_age_s:.0f}s",
                    "OK": "✓" if g.is_tradeable else g.liquidity_reason or "✗",
                })
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            else:
                st.info("Sin griegas parseables.")

    _live_panel()


# ── Footer manual ─────────────────────────────────────────────────────────────

def _data_footer() -> None:
    """Pie estable: no fuerza reruns ni parpadeos."""
    last_fetch = st.session_state.get("last_fetch_label", "sin cargar")
    st.caption(f"Datos cargados: {last_fetch}")


# ── App principal ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="OptionsDesk GGAL",
        layout="wide",
        page_icon=":chart_with_upwards_trend:",
    )
    _inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("OptionsDesk")

        demo_default = os.environ.get("OPTIONS_DESK_DEFAULT_DEMO", "").lower() == "true"
        demo_mode = st.toggle(
            "Modo demo", value=demo_default or not (settings.is_iol_configured() or settings.is_configured()),
            help="Datos sinteticos. Activa cuando no hay credenciales.",
        )
        if st.button("Actualizar datos", key="btn_refresh_sidebar", use_container_width=True):
            st.session_state["force_refresh"] = True
            st.rerun()
        scalp_profile = st.selectbox(
            "Perfil scalping",
            ["balanced", "conservative", "aggressive"],
            index=0,
            help="balanced es el default para operar manualmente con dinero real.",
        )

        capital_value = float(st.session_state.get("scalp_capital", settings.default_capital or 0) or 0)
        capital = capital_value if capital_value > 0 else None

        directional_on = st.toggle(
            "Contexto tecnico multi-TF",
            value=True,
            help="Ajuste leve basado en tendencia. Mejora con historial acumulado.",
        )

        advanced_mode = st.toggle(
            "Modo avanzado",
            value=False,
            help="Suma tabs de análisis de tasas: Inicio, Oportunidades, Referencia, "
                 "Cadena completa, Simulador P&L e Historial.",
        )

        # Filtros: solo visibles en modo avanzado
        min_spread = float(settings.min_tna_spread_pct)
        price_mode = "mid"
        capital_mode = "net"
        show_all = False
        send_alerts = False

        if advanced_mode:
            st.divider()
            st.subheader("Filtros")
            min_spread = st.slider(
                "Spread min vs caucion (%)", 0.0, 30.0,
                float(settings.min_tna_spread_pct), 0.5,
            )
            price_mode = st.selectbox(
                "Precio opcion", ["mid", "bid", "ask", "last"],
                help="mid = analisis; bid = mas realista al vender",
            )
            capital_mode = st.selectbox(
                "Capital puts", ["net", "gross"],
                help="net = K-prima; gross = K (mas conservador)",
            )
            show_all = st.checkbox("Mostrar toda la cadena sin filtro", False)
            st.divider()
            send_alerts = st.checkbox("Enviar alertas Telegram", False)

        st.divider()
        st.caption("GGAL — BYMA — opciones americanas")
        if demo_mode:
            st.info("Datos sinteticos. Configura .env para datos reales.")

    # ── Providers y scanners ──────────────────────────────────────────────────
    provider = _build_provider(demo_mode)
    expiry_cal = _load_expiry_calendar()

    screener = Screener(ScreenerConfig(min_tna_spread_pct=min_spread))
    alerter = TelegramAlerter()

    # ── Datos: carga inicial + refresh manual, sin parpadeo por auto-rerun ─────
    force_refresh = bool(st.session_state.pop("force_refresh", False))
    provider_key = f"{'demo' if demo_mode else 'live'}:{provider.__class__.__name__}"
    if st.session_state.get("provider_key") != provider_key:
        st.session_state["provider_key"] = provider_key
        force_refresh = True
    if force_refresh or "chain" not in st.session_state or st.session_state.chain is None:
        with st.spinner("Cargando datos de mercado..."):
            st.session_state.chain = provider.get_options_chain()
            st.session_state.caucion_tna = provider.get_caucion_tna() or settings.default_caucion_tna
            st.session_state.last_fetch_label = pd.Timestamp.now().strftime("%H:%M:%S")

    chain: Optional[OptionsChain] = st.session_state.chain
    caucion_tna: float = st.session_state.caucion_tna

    if chain is None:
        st.error("Sin datos de mercado. Verifica la conexion o activa modo demo.")
        return
    expiry_cal = _effective_expiry_calendar(chain, expiry_cal)
    cc_scanner = CoveredCallScanner(CoveredCallConfig(price_mode=price_mode), expiry_cal)
    sp_scanner = ShortPutScanner(
        ShortPutConfig(price_mode=price_mode, capital_mode=capital_mode), expiry_cal
    )

    benchmark = (
        Benchmark(caucion_tna_pct=caucion_tna, days=30)
        if caucion_tna > 0
        else ZERO_BENCHMARK
    )
    spot = chain.spot.mid

    # ── Scan ──────────────────────────────────────────────────────────────────
    cc_all = cc_scanner.scan(chain, benchmark)
    sp_all = sp_scanner.scan(chain, benchmark)
    cc_filtered, sp_filtered = screener.rank(cc_all, sp_all, benchmark)

    # ── Historial del subyacente (para AT + VolEdge) ──────────────────────────
    spot_history_df = _load_spot_history(days=180, allow_synthetic=demo_mode)
    htf_df          = _load_htf_history(weeks=52, allow_synthetic=demo_mode)
    ltf_df          = _load_ltf_history()

    spot_history_list: Optional[list[float]] = (
        spot_history_df["close"].dropna().tolist()
        if spot_history_df is not None and not spot_history_df.empty
        else None
    )

    # ── Contexto de mercado (opcional) ────────────────────────────────────────
    from optionsdesk.signals.directional import compute_market_context
    ctx_data = spot_history_df if spot_history_df is not None else None
    context = compute_market_context(ctx_data, htf=htf_df, ltf=ltf_df)
    recommender_context = context if directional_on else None

    # ── Recomendaciones ───────────────────────────────────────────────────────
    recommender = Recommender()
    recs = recommender.recommend(
        cc_all, sp_all, benchmark, context=recommender_context, capital=capital,
        spot_history=spot_history_list, top_n=3,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.title("OptionsDesk — Scalping GGAL")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("GGAL", f"${spot:,.0f}")
    c2.metric("Caucion TNA", f"{caucion_tna:.1f}%" if caucion_tna > 0 else "—")
    c3.metric("Opciones en cadena", len(chain.options))
    c4.metric("Modo", "Demo" if demo_mode else "Live")
    c5.metric("Hora", pd.Timestamp.now().strftime("%H:%M:%S"), delta_color="off")

    if context is not None:
        st.caption(f"Contexto tecnico: {context.note}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # Operatoria diaria: Scalping, Portfolio y Direccional. El resto (análisis
    # de tasas, tutoriales, simuladores) queda detrás de "Modo avanzado".
    if advanced_mode:
        tab_labels = [
            "Scalping", "Portfolio", "Direccional",
            "Inicio", "Oportunidades", "Referencia",
            "Cadena completa", "Simulador P&L", "Historial",
        ]
        (t_scalping, t_portfolio, t_dir, t_home, t_ops, t_learn,
         t_chain, t_sim, t_hist) = st.tabs(tab_labels)
    else:
        t_scalping, t_portfolio, t_dir = st.tabs(["Scalping", "Portfolio", "Direccional"])

    with t_scalping:
        _tab_scalping(
            chain=chain,
            spot=spot,
            provider=provider,
            context=context,
            expiry_calendar=expiry_cal,
            spot_history_df=spot_history_df,
            capital=capital,
            risk_profile=scalp_profile,
            caucion_tna=caucion_tna,
            ltf_df=ltf_df,
        )

    with t_portfolio:
        _tab_portfolio(provider, chain, spot)

    with t_dir:
        _tab_directional(
            spot_history_df, chain, spot, context,
            htf_df=htf_df, ltf_df=ltf_df,
            expiry_calendar=expiry_cal,
            benchmark=benchmark,
        )

    if advanced_mode:
        with t_home:
            _tab_home(recs, alerter)
        with t_ops:
            _tab_opportunities(
                cc_all, sp_all, cc_filtered, sp_filtered,
                show_all, caucion_tna, send_alerts, alerter,
            )
        with t_learn:
            _tab_learn()
        with t_chain:
            _tab_chain(chain, spot)
        with t_sim:
            _tab_simulator(cc_all, sp_all, spot)
        with t_hist:
            _tab_history()

    # ── Footer manual ─────────────────────────────────────────────────────────
    _data_footer()


if __name__ == "__main__":
    main()