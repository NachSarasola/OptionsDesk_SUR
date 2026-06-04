import sys

with open('optionsdesk/ui/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '    with t_portfolio:\n        _tab_portfolio(provider, chain, spot)'

replacement = '''
    # Unified Adaptive Context
    from optionsdesk.backtest.stock_demo import load_stock_demo_trades
    from optionsdesk.backtest.adaptive import analyze_performance
    from pathlib import Path
    unified_trades = load_stock_demo_trades(Path(settings.stock_demo_trades_file), limit=50)
    adaptive_context = analyze_performance(unified_trades) if unified_trades else None

    with t_portfolio:
        _tab_portfolio(
            provider=provider,
            chain=chain,
            spot=spot,
            cc_filtered=cc_filtered,
            sp_filtered=sp_filtered,
            now=now,
            caucion_tna=caucion_tna,
            adaptive_context=adaptive_context,
        )'''

if target in content:
    new_content = content.replace(target, replacement)
    with open('optionsdesk/ui/dashboard.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Success')
else:
    print('Target not found')
