import pandas as pd
import ast

df = pd.read_csv("data/FLODIS_mortality_damage_clean.csv")

rows = []

for _, row in df.iterrows():
    value = row["GFD_matches_nr"]

    if pd.isna(value):
        continue

    value = str(value).strip()

    if value.startswith("["):
        gfd_ids = ast.literal_eval(value)
    else:
        gfd_ids = [int(value)]

    for gfd_id in gfd_ids:
        new_row = row.copy()
        new_row["GFD_ID"] = int(gfd_id)
        rows.append(new_row)

clean = pd.DataFrame(rows)

clean.to_csv("data/FLODIS_mortality_damage_GFD_ready.csv", index=False)

print(clean[["event_id", "GFD_ID", "ISO3", "year", "total_deaths", "total_damages_(000_USD)"]].head())
print("Rows:", len(clean))