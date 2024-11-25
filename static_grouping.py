import pandas as pd
import numpy as np
import os

# Get files into pandas
dataframes = {}
directory = "static_activation_stats"
files = []
for filename in os.listdir(directory):
    files.append(filename)

for file in files:
    dataframes[file] = pd.read_csv(directory + "/" + file)

# Get min and max for each layer
all_dfs = []
for name, df in dataframes.items():
    temp_df = df[["layer_num", "layer_name", "layer_type", "min", "max"]]
    all_dfs.append(temp_df)

combined_df = pd.concat(all_dfs, ignore_index=True)
summary_df = combined_df.groupby(
    ["layer_num", "layer_name", "layer_type"], as_index=False
).agg(
    min=("min", "min"),
    max=("max", "max")
)

print(summary_df.head())
summary_df.to_csv(os.path.join(directory, f'total_activation_statistics.csv'), index=True)