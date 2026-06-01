import sys
sys.path.append(r"c:\Users\Usuario\Desktop\OptionsDesk")

from optionsdesk.data.providers.byma_open import BymaOpenProvider

def test():
    provider = BymaOpenProvider()
    print("Testing Byma Open Caucion TNA...")
    tna = provider.get_caucion_tna()
    print(f"TNA from BYMA: {tna}")

if __name__ == "__main__":
    test()
