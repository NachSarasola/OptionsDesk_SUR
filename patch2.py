import sys

with open('optionsdesk/ui/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

start_sig = 'def _tab_portfolio(provider, chain, spot):'
start_idx = content.find(start_sig)
if start_idx == -1:
    print('Not found start')
    sys.exit(1)
    
end_sig = '# ── Operatoria real (IOL) ─────────────────────────────────────────────────────'
end_idx = content.find(end_sig, start_idx)
if end_idx == -1:
    print('End not found')
    sys.exit(1)

new_code = '''def _tab_portfolio(provider, chain, spot, cc_filtered=None, sp_filtered=None, now=None, caucion_tna=60.0, adaptive_context=None):
    import streamlit as st
    import pandas as pd
    st.subheader("Opciones: Rentas en Vivo (Demo)")
    try:
        from optionsdesk.backtest.options_demo import run_options_demo_tick
    except ImportError:
        st.error("No se pudo importar options_demo")
        return
        
    candidates = []
    if cc_filtered: candidates.extend(cc_filtered[:3])
    if sp_filtered: candidates.extend(sp_filtered[:3])
    candidates.sort(key=lambda x: getattr(x, "score", 0), reverse=True)
    
    try:
        demo = run_options_demo_tick(
            chain=chain,
            candidates=candidates,
            now=now,
            adaptive_context=adaptive_context,
            caucion_tna_pct=caucion_tna,
        )
        positions = demo.get("positions", [])
        closed = demo.get("closed", [])
    except Exception as e:
        st.error(f"Error corriendo options demo: {e}")
        positions = []
        closed = []

    if not positions:
        st.info("Buscando oportunidades seguras de yield...")
    else:
        st.caption("Posiciones vivas automatizadas (Short Puts / Covered Calls)")
        rows = []
        import datetime
        from dataclasses import asdict
        for p in positions:
            quote = chain.options.get(p.symbol) if chain else None
            mark = float(quote.ask) if quote and quote.ask > 0 else None
            pnl_net = None
            if mark is not None:
                pnl_gross = (p.premium_received - mark) * p.contracts * 100
                pnl_net = pnl_gross
                
            entry_d = p.entry_date
            if isinstance(entry_d, str): entry_d = datetime.date.fromisoformat(entry_d)
            now_d = now.date() if now else datetime.date.today()
            days_remaining = (entry_d + datetime.timedelta(days=p.days_entry) - now_d).days
            
            rows.append({
                "Símbolo": p.symbol,
                "Estrategia": p.strategy,
                "DTE": days_remaining,
                "Strike": p.strike,
                "Lotes": p.contracts,
                "Prima (Entrada)": _scalp_money(p.premium_received, 2),
                "Recompra (Ask)": _scalp_money(mark, 2) if mark is not None else "-",
                "PnL Abierto": _scalp_money(pnl_net, 2) if pnl_net is not None else "-",
            })

        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if closed:
        st.caption("Últimas posiciones cerradas")
        c_rows = []
        from dataclasses import asdict
        for c in closed[-5:]:
            d = c.__dict__ if hasattr(c, '__dict__') else asdict(c)
            c_rows.append(d)
        st.dataframe(pd.DataFrame(c_rows), hide_index=True)


'''

new_content = content[:start_idx] + new_code + content[end_idx:]

with open('optionsdesk/ui/dashboard.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print('Success')
