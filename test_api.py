import sys
sys.path.append(r"c:\Users\Usuario\Desktop\OptionsDesk")

from optionsdesk.config.settings import settings
from optionsdesk.data.providers.iol import IOLProvider

def main():
    print("Iniciando prueba de conexión IOL...")
    print(f"Credenciales configuradas: USER {settings.iol_user}")
    provider = IOLProvider()
    
    try:
        provider.connect()
        print("Conectado exitosamente. Obteniendo datos...")
        spot = provider.get_spot()
        print("Spot GGAL:", spot)
        chain = provider.get_options_chain()
        print("Cantidad de opciones GGAL:", len(chain.options) if chain else 0)
        tna = provider.get_caucion_tna()
        print("TNA Caución:", tna)
    finally:
        provider.disconnect()

if __name__ == "__main__":
    main()
