import sys
sys.path.append(r"c:\Users\Usuario\Desktop\OptionsDesk")

from optionsdesk.data.providers.iol import IOLProvider

def explore_iol():
    provider = IOLProvider()
    provider.connect()
    print("Conectado.")
    
    endpoints_to_try = [
        "bCBA/Titulos/PESOS1D/Cotizacion",
        "bCBA/Titulos/PESOS7D/Cotizacion",
        "bCBA/Titulos/CAUCION1D/Cotizacion",
        "bCBA/Titulos/CAUCION30D/Cotizacion",
        "bCBA/Titulos/ARS1D/Cotizacion"
    ]
    
    for ep in endpoints_to_try:
        try:
            print(f"\nProbando: {ep}")
            res = provider._get(ep)
            if res:
                print("Exito. Respuesta:", str(res)[:500])
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    explore_iol()
