import logging
logging.basicConfig(level=logging.INFO)

from optionsdesk.config.settings import settings
from optionsdesk.ui.dashboard import _build_provider
from optionsdesk.core.instruments import load_expiry_calendar
from optionsdesk.core.greeks_engine import compute_chain_greeks

provider = _build_provider(demo=False)
chain = provider.get_options_chain()
cal = load_expiry_calendar()

if chain:
    print(f"Got {len(chain.options)} options in the chain. Spot: {chain.spot.last}")
    greeks = compute_chain_greeks(chain, cal, 40.0)
    print(f"Computed Greeks for {len(greeks)} options.")
    for sym, greek in list(greeks.items())[:5]:
        iv_val = greek.iv * 100 if greek.iv is not None else 0.0
        print(f"Sym: {sym} -> IV: {iv_val:.2f}% | Delta: {greek.delta:.3f}")
else:
    print("Chain is None")
