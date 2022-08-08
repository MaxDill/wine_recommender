import pandas as pd
import math
from utils import avg
import random

def recommend(input_idx, df_wines, df_reviews, sim_matrix, attributes, rec_nb):
    pos = retrieve_pos(input_idx, df_wines)
    global_wines_idx = list(dict.fromkeys(df_reviews['wine_idx'].tolist()))
    global_wines_pos = [retrieve_pos(idx, df_wines) for idx in global_wines_idx]
    sim_scores = list(enumerate(sim_matrix[pos]))
    sim_scores = [sim_scores[i] for i in global_wines_pos] # remove invalid wines
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if pos in global_wines_pos:
        sim_scores = sim_scores[1:rec_nb + 1]
    else:
        sim_scores = sim_scores[0:rec_nb]
    rec_idx = [retrieve_idx(x[0], df_wines) for x in sim_scores]
    ret = df_wines[df_wines['idx'].isin(rec_idx)][attributes]
    ret['score'] = [x[1] for x in sim_scores]
    return ret

def recommend_taster(input_idx, taster, df_wines, df_reviews, sim_matrix, attributes, rec_nb):
    pos = retrieve_pos(input_idx, df_wines)
    taster_wines_idx = retrieve_taster_idx(taster, df_reviews)
    taster_wines_pos = [retrieve_pos(idx, df_wines) for idx in taster_wines_idx]
    sim_scores = list(enumerate(sim_matrix[pos]))
    sim_scores = [sim_scores[i] for i in taster_wines_pos]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if pos in taster_wines_pos:
        sim_scores = sim_scores[1:rec_nb+1]
    else:
        sim_scores = sim_scores[0:rec_nb]
    rec_idx = [retrieve_idx(x[0], df_wines) for x in sim_scores]
    ret = df_wines[df_wines['idx'].isin(rec_idx)][attributes]
    ret['score'] = [x[1] for x in sim_scores]
    return ret

def recommend_no_merge(input_idx, taster, df_reviews, sim_matrix, attributes, rec_nb):
    pos = retrieve_pos(input_idx, df_reviews)
    taster_reviews_idx = df_reviews[df_reviews['taster_name'] == taster]['idx'].tolist()
    taster_reviews_pos = [retrieve_pos(idx, df_reviews) for idx in taster_reviews_idx]
    sim_scores = list(enumerate(sim_matrix[pos]))
    sim_scores = [sim_scores[i] for i in taster_reviews_pos]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if pos in taster_reviews_pos:
        sim_scores = sim_scores[1:rec_nb + 1]
    else:
        sim_scores = sim_scores[0:rec_nb]
    rec_idx = [retrieve_idx(x[0], df_reviews) for x in sim_scores]
    ret = df_reviews[df_reviews['idx'].isin(rec_idx)][attributes]
    ret['score'] = [x[1] for x in sim_scores]
    return ret

def recommend_taster_2(input_idx, taster, df_wines, df_reviews, sim_matrix, attributes, rec_nb):
    pos = retrieve_pos(input_idx, df_wines)
    taster_wines_idx = retrieve_taster_idx(taster, df_reviews)
    wine_color = df_wines[df_wines['idx'] == input_idx]['type'].tolist()[0]
    if wine_color == 'Red' or wine_color == 'White':
        taster_wines = df_wines[df_wines['idx'].isin(taster_wines_idx)]
        taster_wines = taster_wines[taster_wines['type'] == wine_color]
        taster_wines_idx = taster_wines['idx'].tolist()
    taster_wines_pos = [retrieve_pos(idx, df_wines) for idx in taster_wines_idx]
    sim_scores = list(enumerate(sim_matrix[pos]))
    sim_scores = [sim_scores[i] for i in taster_wines_pos]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    if pos in taster_wines_pos:
        sim_scores = sim_scores[1:rec_nb+1]
    else:
        sim_scores = sim_scores[0:rec_nb]
    rec_idx = [retrieve_idx(x[0], df_wines) for x in sim_scores]
    ret = df_wines[df_wines['idx'].isin(rec_idx)][attributes]
    ret['score'] = [x[1] for x in sim_scores]
    return ret

def base_recommender(taster, df_wines, df_reviews,  sim_matrix, nb_wines=5, nb_rec=10):
    rec_idx_list = []
    rec_names_list = []
    wines_scores = get_taster_wines(df_reviews, taster, True)
    wines_scores = wines_scores[0:nb_wines]
    tot_score = sum([j for i,j in wines_scores])
    for wine_idx, score in wines_scores:
        wine_weight = math.floor((score/tot_score)*nb_rec)
        cur_rec = recommend(wine_idx, df_wines, sim_matrix, ['idx', 'name'], wine_weight)
        rec_idx_list.append(cur_rec['idx'].tolist())
        rec_names_list.append(cur_rec['name'].tolist())
    rec_idx_list = [item for sublist in rec_idx_list for item in sublist]
    rec_names_list = [item for sublist in rec_names_list for item in sublist]
    return pd.DataFrame({'idx':rec_idx_list, 'name':rec_names_list})

def evaluate_global_score(wine_idx, df_wines, df_reviews, sim_matrix, k=5, weightedAvg=True):
    rec = recommend(wine_idx, df_wines, df_reviews, sim_matrix, ['idx'], k)
    idx_list = rec['idx'].tolist()
    avg_list = []
    for wine_idx in idx_list:
        score_avg = get_avg_score(wine_idx, df_reviews)
        avg_list.append(score_avg)
    if weightedAvg == True:
        return avg(avg_list, weights=rec['score'].tolist())
    return avg(avg_list)

def evaluate_random_score():
    return random.random()

def evaluate_taster_score(taster, wine_idx, df_wines, df_reviews, sim_matrix, k=5, weightedAvg=True, colorRestriction = False):
    rec = None
    if colorRestriction == False:
        rec = recommend_taster(wine_idx, taster, df_wines, df_reviews, sim_matrix, ['idx'], k)
    else:
        rec = recommend_taster_2(wine_idx, taster, df_wines, df_reviews, sim_matrix, ['idx'], k)
    idx_list = rec['idx'].tolist()
    avg_list = []
    for wine_idx in idx_list:
        score_avg = get_avg_score(wine_idx, df_reviews)
        avg_list.append(score_avg)
    if weightedAvg == True:
        return avg(avg_list, weights=rec['score'].tolist())
    return avg(avg_list)

def evaluate_no_merge_score(taster, review_idx, df_reviews, sim_matrix, k=5, weightedAvg=True):
    rec = recommend_no_merge(review_idx, taster, df_reviews, sim_matrix, ['idx'], k)
    idx_list = rec['idx'].tolist()
    avg_list = []
    for rev_idx in idx_list:
        score_avg = df_reviews[df_reviews['idx'] == rev_idx]['points'].tolist()[0]
        avg_list.append(score_avg)
    if weightedAvg == True:
        return avg(avg_list, weights=rec['score'].tolist())
    return avg(avg_list)


def get_taster_wines(df_reviews, taster, sort):
    ret = []
    taster_reviews = df_reviews[df_reviews['taster_name'] == taster]
    for row in taster_reviews.itertuples(index=False):
        wine_idx = getattr(row, 'wine_idx')
        if wine_idx is not None:
            ret.append((wine_idx, getattr(row, 'points')))
    if sort:
        ret = sorted(ret, key=lambda tup: tup[1], reverse=True)
    return ret

def get_avg_score(wine_idx, df_reviews):
    return df_reviews[df_reviews['wine_idx'] == wine_idx]['points'].mean()

def retrieve_idx(pos, df):
    as_list = df['idx'].tolist()
    i = 0
    for cur_idx in as_list:
        if i == pos:
            return cur_idx
        i = i + 1
    return -1

def retrieve_pos(idx, df):
    as_list = df['idx'].tolist()
    i = 0
    for cur_idx in as_list:
        if cur_idx == idx:
            return i
        i = i + 1
    return -1

def retrieve_taster_idx(taster, df_reviews):
    filter = df_reviews[df_reviews['taster_name'] == taster]
    idx_list = filter['wine_idx'].tolist()
    idx_list = list(dict.fromkeys(idx_list))
    return idx_list