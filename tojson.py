import pandas as pd

df = pd.read_parquet("outputs/m3.parquet")

df.to_json("outputs/m3.json", orient="records", force_ascii=False, indent=2)