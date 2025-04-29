# Updates dep, el, mt, and pos experiment CSVs with URIEL+ distances

import pandas as pd
import os

distances = ["GENETIC", "SYNTACTIC", "FEATURAL", "PHONOLOGICAL", "INVENTORY", "GEOGRAPHIC"]

# Update dep experiment csv with URIEL+ distances
# Load dep experiment csv and dep distance csv
dep_df = pd.read_csv("experiment_csvs//URIEL//dep.csv")
dep_distances_df = pd.read_csv("distances//dep_distances.csv")

# Replace the distances in dep experiment csv with those from dep distances
dep_df[distances] = dep_distances_df[distances]
os.makedirs('src//csv_datasets', exist_ok=True)
dep_df.to_csv("src//csv_datasets//dep.csv", index=False)


# Update el experiment csv with URIEL+ distances
# Load el experiment csv and el distance csv
el_df = pd.read_csv("experiment_csvs//URIEL//el.csv")
el_distances_df = pd.read_csv("distances//el_distances.csv")

# Replace the distances in el experiment csv with those from el distances
el_df[distances] = el_distances_df[distances]
el_df.to_csv("src//csv_datasets//el.csv", index=False)


# Update mt experiment csv with URIEL+ distances
# Load mt experiment csv and mt distance csv
mt_df = pd.read_csv("experiment_csvs//URIEL//mt.csv")
mt_distances_df = pd.read_csv("distances//mt_distances.csv")

# Replace the distances in mt experiment csv with those from mt distances
mt_df[distances] = mt_distances_df[distances]
mt_df.to_csv("src//csv_datasets//mt.csv", index=False)


# Update pos experiment csv with URIEL+ distances
# Load pos experiment csv and pos distance csv
pos_df = pd.read_csv("experiment_csvs//URIEL//pos.csv")
pos_distances_df = pd.read_csv("distances//pos_distances.csv")

# Replace the distances in pos experiment csv with those from pos distances
pos_df[distances] = mt_distances_df[distances]
pos_df.to_csv("src//csv_datasets//pos.csv", index=False)