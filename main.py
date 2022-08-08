import pandas as pd
import pickle
from data_processing import *
from statistics import mean
from sklearn.metrics.pairwise import cosine_similarity
from content_recommender import *
from model_evaluation import *
#from plot import *

def main():
    #df_reviews = pd.read_csv("wine_dataset_1/winemag-data-130k-v2.csv")
    df_reviews = pd.read_pickle("df_reviews_filtered.pkl")
    #df_reviews = df_reviews[df_reviews['taster_name'].notna()]
    #df_reviews.drop(df_reviews.columns[[0]], axis=1, inplace=True)
    #df_reviews = rem_accents(df_reviews, ['title', 'winery', 'variety'])
    #df_reviews.to_pickle('df_reviews.pkl')

    #df_wines = process_df("df_wines.pkl")
    df_wines = pd.read_pickle("df_wines_filtered.pkl")

    df_wines = rem_nan(df_wines, ['year', 'price'])
    #df_quiz = pd.read_csv("quiz.csv")

    #inter = intersect(df_wines['producer'], df_reviews['winery'])
    #with open('winery_inter.pkl', 'wb') as f:
    #    pickle.dump(inter, f)
    with open('winery_inter.pkl', 'rb') as f:
        winery_inter = pickle.load(f)

    df_reviews = df_reviews[df_reviews['winery'].isin(winery_inter.keys())]
    df_wines = df_wines[df_wines['producer'].isin(winery_inter.keys())]

    df_wines = extract_designation(df_wines, 'name', 'producer', 'varieties', 'designation')
    #df_reviews = rem_accents(df_reviews, ['designation'])
    df_reviews = rem_accents(df_reviews, ['variety'])
    df_reviews = rem_engl_words(df_reviews)

    #df_reviews.to_pickle('df_reviews_threshold.pkl')
    #df_wines.to_pickle('df_wines_threshold.pkl')

    sim = similiraty(df_wines, df_reviews, winery_inter)

    sim = sim_threshold(sim, 0.99, 0.99, rem_empty=True)

    # remove reviews assigned at multiple wines
    sim = remove_duplicates_wines(sim)

    df_reviews_no_merge, df_reviews_no_md = create_reviews_no_merge(df_reviews)
    cos_sim_df_reviews = cosine_similarity(df_reviews_no_md, df_reviews_no_md)

    #rec = recommend_no_merge(12, 'Virginie Boone', df_reviews_no_merge, cos_sim_df_reviews, ['idx', 'taster_name', 'description'], 5)
    rec = evaluate_no_merge_score('Virginie Boone', 12, df_reviews_no_merge, cos_sim_df_reviews, k=5, weightedAvg=True)

    df_reviews = add_wines_idx(df_reviews, sim)
    df_reviews = df_reviews[df_reviews['wine_idx'].notnull()]
    df_reviews = normalize_col(df_reviews, 'points')

    #df_reviews.to_pickle('df_reviews_filtered2.pkl')


    df_wines_filtered = df_wines[df_wines['idx'].isin(list(dict.fromkeys(df_reviews['wine_idx'].tolist())))]
    #df_wines_filtered.drop(21247, axis=0, inplace=True) # Drop the only entry that miss sweet feature
    #df_reviews = df_reviews[df_reviews['wine_idx'] != 21247]
    #df_wines_filtered = extended_features_encoding(df_wines_filtered)
    df_wines_filtered = numerize_cols(df_wines_filtered, ['sweet', 'acidity', 'body', 'tannin'], ['SWEET', 'ACIDITY', 'BODY', 'TANNIN'])
    df_wines_no_md = df_wines_filtered .loc[:, ~df_wines_filtered .columns.isin(['name', 'producer', 'use', 'abv', 'degree', 'price', 'year', 'ml', 'local', 'varieties', 'idx', 'designation'])]
    df_wines_no_md = df_wines_filtered[['sweet', 'acidity', 'body', 'tannin']]

    cos_sim = cosine_similarity(df_wines_no_md, df_wines_no_md)

    #ks = [1]
    ks = [1, 3, 5, 10, 15]

    max_iter = 5
    tasters = list(df_reviews['taster_name'].value_counts().index)[0:10]
    #tasters = list(df_reviews_no_merge['taster_name'].value_counts().index)[0:13]
    splits = 5
    #validate = validate_model_rmse(tasters, ks, max_iter, splits, df_wines_filtered, df_reviews_no_merge, cos_sim_df_reviews)
    validate = validate_model_rmse(tasters, ks, max_iter, splits, df_wines_filtered, df_reviews, cos_sim)
    #random_des =  select_random_inter(sim, 0.1, df_wines, df_reviews)
    #plot_distribution([vi[2] for k,v in sim.items() for vi in v], 5, 'Distribution of jaro-winkler similarities for wine varieties', 'Y')
    #plot_scatter_taster_rmse()
    print('Breakpoint')

def process_df(file_name):
    """
    Processes the cleansingswine/wine_info.csv such that it becomes usable for making a recommender system.
    :param file_name: the name of the output file containing the possessed dataframe (if None, no file is created)
    :return: the processed dataframe
    """
    df = pd.read_csv("cleansingswine/wine_info.csv")
    df = col_to_list(df, 'use', ', ')
    df = abv_degree_averaging(df, ['abv', 'degree'])
    df = ASCIIfy_df(df, ["producer", 'nation', 'local1', 'local2', 'local3', 'local4'])
    df = merge_cols(df, ["local1", "local2", "local3", "local4"], "local", True, True)
    df = merge_cols(df,
                    ["varieties1", "varieties2", "varieties3", "varieties4", "varieties5", "varieties6", "varieties7",
                     "varieties8", "varieties9", "varieties10", "varieties11", "varieties12"], "varieties", True, True)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    if file_name != None:
        df.to_pickle(file_name)
    return df

#country, variety, price, province
def create_reviews_no_merge(df_reviews):
    new_df = df_reviews.copy()
    new_df = new_df[new_df['taster_name'].notna()]
    new_df = new_df[new_df['price'].notna()]
    new_df = new_df[new_df['province'].notna()]
    new_df = new_df[new_df['variety'].notna()]
    new_df = normalize_col(new_df, 'points')
    new_df2 = new_df.copy()
    new_df = new_df[['price', 'province', 'variety']]
    one_hot_1 = pd.get_dummies(new_df['province'], prefix='province')
    one_hot_2 = pd.get_dummies(new_df['variety'], prefix='variety')
    new_df = new_df.drop('province', axis=1)
    new_df = new_df.drop('variety', axis=1)
    new_df = new_df.join(one_hot_1)
    new_df = new_df.join(one_hot_2)
    return new_df2, new_df

if __name__ == "__main__":
    main()