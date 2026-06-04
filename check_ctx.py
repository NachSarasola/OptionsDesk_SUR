import asyncio
from optionsdesk.ui.dashboard import _load_spot_history, _load_htf_history, _load_ltf_history
from optionsdesk.signals.directional import compute_market_context

def check_context():
    spot_history_df = _load_spot_history(days=180)
    htf_df = _load_htf_history(weeks=52)
    ltf_df = _load_ltf_history()
    
    print('spot_history_df empty:', spot_history_df.empty if spot_history_df is not None else True)
    print('htf_df empty:', htf_df.empty if htf_df is not None else True)
    print('ltf_df empty:', ltf_df.empty if ltf_df is not None else True)
    
    context = compute_market_context(spot_history_df, htf=htf_df, ltf=ltf_df)
    if context is None:
        print('Context is None')
    else:
        print(f'Context: trend={context.trend}, confidence={context.confidence}, note={context.note}')
        snap = getattr(context, 'ltf_snap', None) or getattr(context, 'snap', None)
        if snap is None:
            print('snap is None (Intraday context missing)')
        else:
            print(f'snap found: momentum={getattr(snap, "momentum", "N/A")}')

if __name__ == '__main__':
    check_context()
