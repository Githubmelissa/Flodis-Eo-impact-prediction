import pandas as pd

# load your original file
df = pd.read_csv("data/FLODIS_mortality_damage.csv")

# rename the empty column
df = df.rename(columns={"Unnamed: 0": "event_id"})

# save cleaned file
df.to_csv("data/FLODIS_mortality_damage_clean.csv", index=False)

# check result
print(df.columns[:10])