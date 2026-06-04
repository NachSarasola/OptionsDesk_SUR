import sys
from datetime import datetime
from optionsdesk.config.settings import settings
from optionsdesk.data.providers.base import MarketDataProvider
from optionsdesk.signals.scalping import scan_all, build_scalp_verdict
from optionsdesk.ui.dashboard import _build_provider, _load_spot_history, _load_ltf_history, _effective_expiry_calendar

def verify():
    print("Iniciando verificación...")
    provider = _build_provider()
    health = provider.get_health()
    print(f"Provider: {health.source}")
    print(f"Realtime: {health.source in {'PRIMARY', 'IOL', 'HOMEBROKER'} and health.connected}")
    print(f"Connected: {health.connected}")
    print(f"Timeouts: {health.timeouts}")
    
    chain = provider.get_options_chain()
    if chain is None:
        print("Chain is None. No data from provider.")
        return

    print(f"Spot: {chain.spot.mid}")
    
    capital = float(settings.default_capital or 0)
    print(f"Capital config (.env DEFAULT_CAPITAL): {capital}")
    
    # Intraday
    ltf_df = _load_ltf_history()
    print(f"LTF History empty: {ltf_df.empty if ltf_df is not None else True}")
    
    # Scanner context
    from optionsdesk.signals.technical import analyze
    snap_now = analyze(ltf_df) if ltf_df is not None and not ltf_df.empty else None
    print(f"Technical Context available: {snap_now is not None}")

    # Pulso
    from optionsdesk.signals.scalping import compute_spot_tape_pulse
    pulse = compute_spot_tape_pulse([]) # Empty pulse because it's a dry run
    print(f"Pulse Confirmed: {pulse.confirmed}")

    expiry_cal = {}
    
    # Test scan
    try:
        greeks, signals = scan_all(
            chain=chain,
            expiry_calendar=_effective_expiry_calendar(chain, expiry_cal),
            snap=snap_now,
            realized_vol=0.3,
            r=0.0,
            capital=capital if capital > 0 else 500000, # Fake capital to test other blockers
            positions=[],
            risk_profile="balanced",
            allow_overnight_entries=False,
            require_live_confirmation=False,
            live_confirmation=True, # Pretend it's live
            live_direction=pulse.direction
        )
        print(f"Signals generated: {len(signals)}")
        actionables = [s for s in signals if s.confidence == "actionable"]
        print(f"Actionable Signals: {len(actionables)}")
        
        if not actionables and signals:
            print("Top signals and their blockers:")
            for s in sorted(signals, key=lambda x: x.score, reverse=True)[:5]:
                print(f" - {s.symbol} Score {s.score}: {s.confidence}")
                print(f"   Blockers: {getattr(s, 'blocking_flags', []) or s.risk_flags}")
    except Exception as e:
        print(f"Error scanning: {e}")

if __name__ == '__main__':
    verify()
