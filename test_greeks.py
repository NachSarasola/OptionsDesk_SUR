from optionsdesk.data.providers.demo import DemoProvider
from optionsdesk.core.instruments import load_expiry_calendar
from optionsdesk.signals.scalping import scan_all

provider = DemoProvider()
chain = provider.get_options_chain()
cal = load_expiry_calendar()

try:
    greeks, signals = scan_all(chain, cal, snap=None, smile_map=None, realized_vol=None)
    print("Greeks:", len(greeks))
    print("Signals:", len(signals))
except Exception as e:
    import traceback
    traceback.print_exc()
