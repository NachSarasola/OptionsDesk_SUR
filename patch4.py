import sys

with open('optionsdesk/ui/dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    # Unified Adaptive Context
    from optionsdesk.backtest.stock_demo import load_stock_demo_trades
    from optionsdesk.backtest.adaptive import analyze_performance
    from pathlib import Path
    unified_trades = load_stock_demo_trades(Path(settings.stock_demo_trades_file), limit=50)
    adaptive_context = analyze_performance(unified_trades) if unified_trades else None'''

replacement = '''    # Unified Adaptive Context
    from optionsdesk.backtest.stock_demo import load_stock_demo_trades
    from optionsdesk.backtest.adaptive import analyze_performance
    from pathlib import Path
    import json
    
    unified_trades = load_stock_demo_trades(Path(settings.stock_demo_trades_file), limit=50) if Path(settings.stock_demo_trades_file).exists() else []
    
    options_file = Path(settings.app_dir) / "options_demo_trades.jsonl"
    if options_file.exists():
        try:
            with options_file.open("r", encoding="utf-8") as fh:
                from dataclasses import make_dataclass
                # Create a generic object to hold pnl_ars so analyze_performance can read it
                class DummyTrade:
                    def __init__(self, pnl):
                        self.pnl_ars = pnl
                        
                for line in fh:
                    if line.strip():
                        data = json.loads(line)
                        if "pnl_ars" in data:
                            unified_trades.append(DummyTrade(data["pnl_ars"]))
        except Exception as e:
            pass
            
    adaptive_context = analyze_performance(unified_trades) if unified_trades else None'''

if target in content:
    new_content = content.replace(target, replacement)
    with open('optionsdesk/ui/dashboard.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Success')
else:
    print('Target not found')
