import sys

def replace_in_file(filepath, target, replacement):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if target in content:
        content = content.replace(target, replacement)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed {filepath}")
    else:
        print(f"Target not found in {filepath}")

# Fix options_demo.py
replace_in_file(
    'optionsdesk/backtest/options_demo.py',
    'Path(settings.app_dir) / "options_demo_positions.json"',
    'Path("data/options_demo_positions.json")'
)

replace_in_file(
    'optionsdesk/backtest/options_demo.py',
    'Path(settings.app_dir) / "options_demo_trades.jsonl"',
    'Path("data/options_demo_trades.jsonl")'
)

# Fix dashboard.py
replace_in_file(
    'optionsdesk/ui/dashboard.py',
    'Path(settings.app_dir) / "options_demo_trades.jsonl"',
    'Path("data/options_demo_trades.jsonl")'
)
