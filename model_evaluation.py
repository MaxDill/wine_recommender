import pandas as pd
import random
import math
import main
from statistics import mean, stdev
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from content_recommender import evaluate_taster_score, evaluate_global_score, evaluate_random_score, evaluate_no_merge_score

#tast_dict_pred = {}
#tast_dict_true = {}

def evaluate_model_rmse(tasters, df_wines, df_reviews, sim_matrix, k, splits_nb=5, seed=42):
    all_tasters_true_scores = []
    all_tasters_pred_scores = []
    for taster in tasters:
        # if taster not in tast_dict_pred:
        #     tast_dict_pred[taster] = []
        #     tast_dict_true[taster] = []
        df_reviews_taster = df_reviews[df_reviews['taster_name'] == taster]
        kf = KFold(n_splits=splits_nb, shuffle=True, random_state=seed)
        #df_reviews_train, df_reviews_test = train_test_split(df_reviews_taster, test_size=test_size)
        for split in kf.split(df_reviews_taster):
            df_reviews_train = df_reviews_taster.iloc[split[0]]
            df_reviews_test = df_reviews_taster.iloc[split[1]]
            test_idx = df_reviews_test['wine_idx'].tolist()
            #test_idx = df_reviews_test['idx'].tolist()
            true_scores = df_reviews_test['points'].tolist()
            pred_scores = []
            for wine_idx in test_idx:
                pred_score = evaluate_taster_score(taster, wine_idx, df_wines, df_reviews_train, sim_matrix, k=k, weightedAvg=True, colorRestriction=False)
                #pred_score = evaluate_no_merge_score(taster, wine_idx, df_reviews_train, sim_matrix, k=k, weightedAvg=True)
                #pred_score = evaluate_global_score(wine_idx, df_wines, df_reviews_train, sim_matrix, k=k, weightedAvg=True)
                #pred_score = evaluate_random_score()
                pred_scores.append(pred_score)
            all_tasters_true_scores.extend(true_scores)
            all_tasters_pred_scores.extend(pred_scores)
            #tast_dict_true[taster].extend(true_scores)
            #tast_dict_pred[taster].extend(pred_scores)
    return mean_squared_error(all_tasters_true_scores, all_tasters_pred_scores, squared=False)

def validate_model_rmse(tasters, ks, max_iter, splits_nb, df_wines_filtered, df_reviews, cos_sim):
    all_rmse = []
    for k in ks:
        k_rmse = []
        for iter in range(max_iter):
            rmse = evaluate_model_rmse(tasters, df_wines_filtered, df_reviews, cos_sim, k, splits_nb=splits_nb, seed=42+iter)
            print(rmse)
            k_rmse.append(rmse)
        print('######')
        print(f'k = {k} : {k_rmse}')
        print('######')
        #all_rmse.append(mean(k_rmse))
        all_rmse.append((mean(k_rmse), stdev(k_rmse)))
    return all_rmse

def select_random_inter(sim, ratio, df_wines, df_reviews, seed=42):
    pair_list = []
    for key in sim:
        l = sim[key]
        for rev_idx, prob1, prob2 in l:
            pair_list.append((key, rev_idx))
    random.seed(seed)
    sample = random.sample(pair_list, math.ceil(len(pair_list)*ratio))
    wine_des = []
    rev_des = []
    for idx_wine, idx_rev in sample:
        wine = df_wines[df_wines['idx'] == idx_wine]['designation'].tolist()[0]
        winery = df_wines[df_wines['idx'] == idx_wine]['producer'].tolist()[0]
        rev = df_reviews[df_reviews['idx'] == idx_rev]['designation'].tolist()[0]
        wine_des.append(f'{winery}, {wine}')
        rev_des.append(f'{winery}, {rev}')
    return pd.DataFrame({'Wine designation': wine_des, 'Review designation': rev_des})
