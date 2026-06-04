from pyobd import BymaData
for sym in ["GGAL", "GGAL 24HS", "GGAL 48HS", "GFG"]:
    try:
        df = BymaData().get_intraday_history(sym)
        print(f"{sym}: empty={df.empty if df is not None else True}, type={type(df)}")
    except Exception as e:
        print(f"{sym}: error {e}")
