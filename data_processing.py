from math import isnan
from statistics import mean

import pandas as pd
import unidecode
from jellyfish import jaro_winkler_similarity
from utils import *

def abv_degree_averaging(df, cols):
    new_df = df.copy()
    for col in cols:
        for idx, item in enumerate(new_df[col]):
            if not (type(item) != str and isnan(item)):
                new_df[col][idx] = abv_degree_converter(item)
    return new_df

def abv_degree_converter(s):
    new_item = s.strip()
    new_item = new_item.split('~')
    new_item = list(map(float, new_item))
    return mean(new_item)

def col_to_list(df, col, split_str):
    new_df = df.copy()
    for idx, item in enumerate(new_df[col]):
        if not (type(item)!=str and isnan(item)):
            new_df[col][idx] = item.split(split_str)
    return new_df

def merge_cols(df, cols, merged_col_name, drop_cols, ignore_nan):
    merged_col = []
    for i in range(0,len(df)):
        merged_col.append([])
    for col in cols:
        for idx, item in enumerate(df[col]):
            if not (ignore_nan and type(item)!=str and isnan(item)):
                merged_col[idx].append(item)
    new_df = df.copy()
    if drop_cols:
        new_df.drop(cols, inplace=True, axis=1)
    new_df[merged_col_name] = merged_col
    return new_df

def ASCIIfy_df(df, cols):
    new_df = df.copy()
    for col in cols:
        for idx, item in enumerate(new_df[col]):
            new_df[col][idx] = ASCIIfy(item)
    return new_df

def ASCIIfy(s):
    if type(s)==str:
        return s.encode("ascii", "ignore").decode().lstrip(' ')
    else:
        return float("NaN")

def rem_accents(df, cols):
    new_df = df.copy()
    for col in cols:
        as_list = df[col].tolist()
        for idx, item in enumerate(as_list):
            if not (type(item)!=str and isnan(item)):
                as_list[idx] = unidecode.unidecode(item)
        new_df[col] = as_list
    return new_df

def rem_engl_words(df):
    new_df = df.copy()
    as_list = df['variety'].tolist()
    for idx, item in enumerate(as_list):
        if not (type(item) != str and isnan(item)):
            as_list[idx] = item.replace('white', 'blanc')
            as_list[idx] = as_list[idx].replace('red', 'rouge')
            as_list[idx] = item.replace('White', 'Blanc')
            as_list[idx] = as_list[idx].replace('Red', 'Rouge')
    new_df['variety'] = as_list
    return new_df

def numerize_cols(df, cols, values):
    new_df = df.copy()
    for col_idx, col in enumerate(cols):
        as_list = new_df[col].tolist()
        for idx, item in enumerate(as_list):
            print(idx, item)
            new_item = int(item.replace(values[col_idx], ''))
            as_list[idx] = new_item
        new_df[col] = as_list
    return new_df

def extended_features_encoding(df_wines):
    new_df = df_wines.copy()
    one_hot_1 = pd.get_dummies(new_df['nation'], prefix='nation')
    one_hot_2 = pd.get_dummies(new_df['type'], prefix='type')
    new_df = new_df.drop('nation', axis=1)
    new_df = new_df.drop('type', axis=1)
    new_df = new_df.join(one_hot_1)
    new_df = new_df.join(one_hot_2)
    return new_df

def rem_nan(df, cols):
    new_df = df.copy()
    for col_idx, col in enumerate(cols):
        indexes = list(new_df.index.values)
        for idx, item in enumerate(new_df[col]):
            if type(item) == float and isnan(item):
                new_df.drop(indexes[idx], axis=0, inplace=True)
    return new_df

def extract_designation(df, designation_col, producer_col, variety_col, new_col_name):
    new_df = df.copy()
    des_as_list = new_df[designation_col].tolist()
    prod_as_list = new_df[producer_col].tolist()
    var_as_list = new_df[variety_col].tolist()
    designations = []
    for idx, name in enumerate(des_as_list):
        split_str = name.split(', ')
        if len(split_str) == 1:
            if name == prod_as_list[idx]:
                designations.append(float("NaN"))
            elif prod_as_list[idx] in name:
                designations.append(name.replace((prod_as_list[idx]+' '), ''))
            else:
                designations.append(name)
        else:
            if split_str[0] == prod_as_list[idx]:
                designations.append(split_str[1])
            else:
                designations.append(split_str[0]+' '+split_str[1])
        if not (type(designations[idx])!=str and isnan(designations[idx])):
            var_list = var_as_list[idx]
            for variety in var_list:
                if variety in designations[idx]:
                    designations[idx] = designations[idx].replace(variety, '')
                    if designations[idx] == '':
                        designations[idx] = float('NaN')
                    break
    new_df[new_col_name] = designations
    return new_df

def normalize_col(df, col):
    new_df = df.copy()
    norm_col =  (df[col]-df[col].min())/(df[col].max()-df[col].min())
    new_df[col] = norm_col
    return new_df

def intersect(col1, col2, limit=None):
    ret = {}
    for idx1, elem1 in enumerate(col1.tolist()):
        if elem1 not in ret:
            for idx2, elem2 in enumerate(col2.tolist()):
                if elem1 == elem2:
                    if elem1 in ret:
                        ret[elem1].append(idx2)
                    else:
                        ret[elem1] = [idx2]
        if idx1 == limit:
            break
    return ret

def add_wines_idx(df, sim):
    new_df = df.copy()
    new_df['wine_idx'] = None
    for wine_idx in sim:
        for review in sim[wine_idx]:
            new_df.loc[new_df['idx'] == review[0], ['wine_idx']] = wine_idx
    return new_df

def similiraty(df_wines, df_reviews, inter):
    ret = {}
    wines = df_wines.loc[:, ['producer','designation', 'varieties', 'idx']]
    reviews = df_reviews.loc[:, ['designation','variety', 'idx']]
    for row_wines in wines.itertuples(index=False):
        wines_idx = getattr(row_wines, 'idx')
        wines_des = getattr(row_wines, 'designation')
        wines_var_list = getattr(row_wines, 'varieties')
        prod = getattr(row_wines, 'producer')
        idx_in_reviews = inter[prod]
        reviews_match = reviews[reviews['idx'].isin(idx_in_reviews)]
        for row_reviews in reviews_match.itertuples(index=False):
            reviews_idx = getattr(row_reviews, 'idx')
            reviews_des = getattr(row_reviews, 'designation')
            reviews_var = getattr(row_reviews, 'variety')
            if type(wines_des) != str and isnan(wines_des):
                wines_des = 'nan'
            if type(reviews_des) != str and isnan(reviews_des):
                reviews_des = 'nan'
            sim_des = jaro_winkler_similarity(wines_des, reviews_des)
            sim_var = -1
            if len(wines_var_list) > 0:
                for wines_var in wines_var_list:
                    cur_sim_var = jaro_winkler_similarity(wines_var, reviews_var)
                    if cur_sim_var > sim_var:
                        sim_var = cur_sim_var
            if wines_idx in ret:
                ret[wines_idx].append((reviews_idx, sim_des, sim_var))
            else:
                ret[wines_idx] = [(reviews_idx, sim_des, sim_var)]
    return ret

def sim_threshold(sim_dict, th1, th2, rem_empty=False):
    ret = {}
    i = 0
    for key in sim_dict:
        candidates_list = sim_dict[key]
        new_list = []
        for candidate in candidates_list:
            idx, sim1, sim2 = candidate
            if sim1 >= th1 and sim2 >= th2:
                new_list.append(candidate)
            if sim1 >= th1 and sim2 < th2:
                i = i + 1
        if not (rem_empty and len(new_list) < 1):
            ret[key] = new_list
    return ret

def retrieve_idx(sim, wine_idx):
    l = sim[wine_idx]
    return [item[0] for item in l]

def dup_filtering(dict):
    new_dict = {}
    for key in dict:
        best_tuple = (None, -1, -1)
        for tuple in dict[key]:
            if tuple[1] > best_tuple[1]:
                best_tuple = tuple
            if tuple[1] == best_tuple[1]:
                if tuple[2] > best_tuple[2]:
                    best_tuple = tuple
        if key in new_dict:
            new_dict[key].append(best_tuple)
        else:
            new_dict[key] = [best_tuple]
    return new_dict

def remove_duplicates_wines(sim):
    rev_dict = reverse_dict(sim)
    rev_dict = dup_filtering(rev_dict)
    rev_dict = reverse_dict(rev_dict)
    return rev_dict
