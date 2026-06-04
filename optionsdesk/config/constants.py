# Constantes de referencia para la estrategia GGAL.
# Calibrar periodicamente con la vol realizada del ultimo año.

# Volatilidad anual de referencia para GGAL cuando no hay IV disponible.
# Se usa como fallback en spreads.py y vertical_spread.py cuando IV = None.
DEFAULT_SIGMA_GGAL: float = 0.65
