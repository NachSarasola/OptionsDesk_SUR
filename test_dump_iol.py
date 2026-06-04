import logging
logging.basicConfig(level=logging.DEBUG)

from optionsdesk.ui.dashboard import _build_provider
import json

provider = _build_provider(demo=False)

paneles = [
    "Cotizaciones/paneles/Opciones/argentina",
    "Cotizaciones/Opciones/argentina",
    "Cotizaciones/opciones/general/argentina"
]

for panel in paneles:
    print(f"Testing {panel}...")
    data = provider._get(panel)
    if data and isinstance(data, list) and len(data) > 0:
        print(f"Success! Panel {panel} returned {len(data)} options.")
        print(json.dumps(data[0], indent=2))
        break
    else:
        print(f"Failed or empty.")
