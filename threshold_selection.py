import pandas as pd
import pickle
from data_processing import *
from model_evaluation import *
df_reviews = pd.read_pickle("df_reviews_threshold.pkl")
df_wines = pd.read_pickle("df_wines_threshold.pkl")
with open('winery_inter.pkl', 'rb') as f:
    winery_inter = pickle.load(f)

seed = 42
thresholds = [0.1, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
res = []

for th in thresholds:
    sim = similiraty(df_wines, df_reviews, winery_inter)
    sim = sim_threshold(sim, th, 0.99, rem_empty=True)
    sim = remove_duplicates_wines(sim)

    new_df_reviews = add_wines_idx(df_reviews, sim)
    new_df_reviews = new_df_reviews[new_df_reviews['wine_idx'].notnull()]

    random_des =  select_random_inter(sim, 1, df_wines, new_df_reviews, seed)
    res.append(random_des[:20])

i = 0
for df in res:
    print(f'Threshold : {thresholds[i]}')
    print(df)
    i = i + 1