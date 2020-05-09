import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

import pickle

def nan_to_tuple(x):
    return x if x else tuple()

def add_dummies_test(df: pd.DataFrame):
    df = df.copy()
    
    mlb = MultiLabelBinarizer()
    with open('train_columns.pkl', 'rb') as f:
        TRAIN_COLUMNS = pickle.load(f)
    # Keywords: (10003 unique)
    dummy_keywords = pd.DataFrame(mlb.fit_transform(df['Keywords.id'].apply(nan_to_tuple)),
                                    columns=[f"keyword_{kw_id}" for kw_id in mlb.classes_], 
                                    index=df.index)
    dummy_keywords = dummy_keywords[TRAIN_COLUMNS['dummy_keywords']]
    df = pd.concat([df, dummy_keywords], axis=1)

    # Genres:
    dummy_genres = pd.DataFrame(mlb.fit_transform(df.genres.apply(nan_to_tuple)),
                            columns=sorted([f"genre_{cl}" for cl in mlb.classes_]), 
                            index=df.index)
    df = pd.concat([df, dummy_genres], axis=1)

    # Companies:
    dummy_companies = pd.DataFrame(mlb.fit_transform(df['production_companies.id'].apply(nan_to_tuple)),
                                columns=[f"company_{cl}" for cl in mlb.classes_], 
                                index=df.index)
    dummy_companies = dummy_companies[TRAIN_COLUMNS['dummy_companies']]
    df = pd.concat([df, dummy_companies], axis=1) # Maybe biggest company size is enough...   

    # Production countries:
    dummy_countries =pd.DataFrame(mlb.fit_transform(df.production_countries.apply(nan_to_tuple)),
                                columns=[f"country_{cl}" for cl in mlb.classes_], 
                                index=df.index)
    dummy_countries = dummy_countries[TRAIN_COLUMNS['dummy_countries']]                                
    df = pd.concat([df, dummy_countries], axis=1)

    # Spoken Languages:
    dummy_lang = pd.DataFrame(mlb.fit_transform(df.spoken_languages.apply(nan_to_tuple)),
                            columns=[f"spoken_lang_{cl}" for cl in mlb.classes_], 
                            index=df.index)
    dummy_lang = dummy_lang[TRAIN_COLUMNS['dummy_lang']]                            
    df = pd.concat([df, dummy_lang], axis=1)

    # Cast:
    # dummy_cast = pd.DataFrame(mlb.fit_transform(df['cast.id'].apply(nan_to_tuple)),
    #                             columns=[f"cast_{cl}" for cl in mlb.classes_], 
    #                             index=df.index)
    # cast_hist = dummy_cast.sum()                                
    # dummy_cast = dummy_cast[TRAIN_COLUMNS['dummy_cast']]      
    # df = pd.concat([df, dummy_cast], axis=1)

    # Original Language dummy:
    dummy_orig_lang = pd.get_dummies(df.original_language, prefix="original_lang")
    dummy_orig_lang = dummy_orig_lang[TRAIN_COLUMNS['dummy_orig_lang']]
    df = pd.concat([df, dummy_orig_lang], axis=1)

    return df


def add_dummies_train(df: pd.DataFrame):
    df = df.copy()

    mlb = MultiLabelBinarizer()
    TRAIN_COLUMNS = dict()

    # Keywords: (10003 unique)
    dummy_keywords = pd.DataFrame(mlb.fit_transform(df['Keywords.id'].apply(nan_to_tuple)),
                                    columns=[f"keyword_{kw_id}" for kw_id in mlb.classes_], 
                                    index=df.index)
    dummy_keywords = dummy_keywords[dummy_keywords.sum().nlargest(20).index]
    df = pd.concat([df, dummy_keywords], axis=1)
    TRAIN_COLUMNS['dummy_keywords'] = dummy_keywords.columns

    # Genres:
    dummy_genres = pd.DataFrame(mlb.fit_transform(df.genres.apply(nan_to_tuple)),
                            columns=sorted([f"genre_{cl}" for cl in mlb.classes_]), 
                            index=df.index)
    df = pd.concat([df, dummy_genres], axis=1)

    # Companies:
    dummy_companies = pd.DataFrame(mlb.fit_transform(df['production_companies.id'].apply(nan_to_tuple)),
                                columns=[f"company_{cl}" for cl in mlb.classes_], 
                                index=df.index)
    dummy_companies = dummy_companies[dummy_companies.sum().nlargest(10).index]
    df = pd.concat([df, dummy_companies], axis=1) # Maybe biggest company size is enough...   
    TRAIN_COLUMNS['dummy_companies'] = dummy_companies.columns

    # Production countries:
    dummy_countries =pd.DataFrame(mlb.fit_transform(df.production_countries.apply(nan_to_tuple)),
                                columns=[f"country_{cl}" for cl in mlb.classes_], 
                                index=df.index)
    dummy_countries = dummy_countries[dummy_countries.sum().nlargest(10).index]                                
    df = pd.concat([df, dummy_countries], axis=1)
    TRAIN_COLUMNS['dummy_countries'] = dummy_countries.columns

    # Spoken Languages:
    dummy_lang = pd.DataFrame(mlb.fit_transform(df.spoken_languages.apply(nan_to_tuple)),
                            columns=[f"spoken_lang_{cl}" for cl in mlb.classes_], 
                            index=df.index)
    dummy_lang = dummy_lang[dummy_lang.sum().nlargest(10).index]                            
    df = pd.concat([df, dummy_lang], axis=1)
    TRAIN_COLUMNS['dummy_lang'] = dummy_lang.columns

    # Cast:
    # dummy_cast = pd.DataFrame(mlb.fit_transform(df['cast.id'].apply(nan_to_tuple)),
    #                             columns=[f"cast_{cl}" for cl in mlb.classes_], 
    #                             index=df.index)
    # cast_hist = dummy_cast.sum()                                
    # dummy_cast = dummy_cast[dummy_cast.sum().nlargest(100).index] # Top 100 actors
    # df = pd.concat([df, dummy_cast], axis=1)
    # TRAIN_COLUMNS['dummy_cast'] = dummy_cast.columns

    # Original Language dummy:
    dummy_orig_lang = pd.get_dummies(df.original_language, prefix="original_lang")
    dummy_orig_lang = dummy_orig_lang[dummy_orig_lang.sum().nlargest(10).index]
    df = pd.concat([df, dummy_orig_lang], axis=1)
    TRAIN_COLUMNS['dummy_orig_lang'] = dummy_orig_lang.columns
    
    with open('train_columns.pkl', 'wb') as f:
        pickle.dump(TRAIN_COLUMNS, f)

    return df


def map_and_max(collection, mapping_dict):
    return max(map(mapping_dict.get, collection)) if collection else None

def eval_or_nan(obj):
    if obj and pd.notnull(obj) and isinstance(obj, str):
        return eval(obj)
    return None

def map_attribute(obj, attribute_name: str):
    if obj:
        iterable = eval(obj) if isinstance(obj, str) else obj
        return tuple(map(lambda x: x.get(attribute_name, None), iterable))
    return None

def smart_len(x, split_char= None):
    if split_char:
        return len(x.split(" ")) if pd.notnull(x) else 0
    return len(x) if pd.notnull(x) else 0

def features_flattening(df: pd.DataFrame):
    df = df.copy()
    df['belongs_to_collection'] = df.belongs_to_collection.apply(eval_or_nan)
    df['belongs_to_collection.id'] = df.belongs_to_collection\
                                            .apply(lambda x: None if pd.isna(x) else x['id']).astype('Int64')


    df['genres'] = df.genres.apply(lambda gs: tuple(g['name'] for g in eval(gs)))

    df['production_companies'] = df.production_companies.apply(eval_or_nan)
    df['production_companies.id'] = df.production_companies\
                                            .apply(lambda companies: map_attribute(companies, 'id'))
    df['production_companies.origin_country'] = df.production_companies\
                                            .apply(lambda companies: map_attribute(companies, 'origin_country'))

    df['production_countries'] = df.production_countries.apply(lambda countries: map_attribute(countries, 'iso_3166_1'))

    df['release_date'] = pd.to_datetime(df.release_date)
    df['release_month'] = df.release_date.dt.month
    df['release_quarter'] = df.release_date.dt.quarter
    df['release_year'] = df.release_date.dt.year

    df['spoken_languages'] = df.spoken_languages.apply(lambda langs: map_attribute(langs, 'iso_639_1'))

    df['Keywords'] = df.Keywords.apply(eval_or_nan)
    df['Keywords.id'] =df.Keywords.apply(lambda keywords: map_attribute(keywords, 'id'))

    df['cast'] = df.cast.apply(eval_or_nan)
    df['cast.id'] = df.cast.apply(lambda actors: map_attribute(actors, 'id'))
    df['cast.gender'] = df.cast.apply(lambda actors: map_attribute(actors, 'gender')) # Gender ratio

    df['crew'] = df.crew.apply(eval)
    df['crew.id'] = df.crew.apply(lambda crew: map_attribute(crew, 'id'))
    df['crew.department'] = df.crew.apply(lambda crew: map_attribute(crew, 'department')) # Dept size
    
    df.drop(['crew', 'cast', 'Keywords', 'belongs_to_collection', 'release_date', 'production_companies'], axis=1, inplace=True)

    return df

def missing_value_imputation(df: pd.DataFrame):
    df = df.copy()

    from sklearn.impute import KNNImputer

    df.budget.fillna(0, inplace=True)
    df.budget.replace(0, -1, inplace= True)
    
    df.runtime.fillna(0, inplace=True)
    df.runtime.replace(0, -1, inplace= True)

    imputer = KNNImputer(missing_values= -1)
    imputed = imputer.fit_transform(df)

    return pd.DataFrame(imputed, columns=df.columns,index=df.index)

def get_element_frequency(df, attribute):
    return Counter(df[attribute].dropna().sum())

# Gender actor ratio: 0 is unspecified, 1 is female, and 2 is male
def genders_ratio(genders):
    arr = np.array(genders)
    males = (arr == 1).sum()
    females = (arr == 2).sum()
    if males or females:
        return males / (females + males)
    return 0

def drop_features(df: pd.DataFrame):
    removed_columns = ['backdrop_path', 'homepage', 'poster_path', 'imdb_id', 'video', 'status']
    return df.drop(columns=removed_columns, axis=1).set_index('id')
    
def feature_extraction(df: pd.DataFrame):
    df = df.copy()

    # Collection size:
    df['collection_size'] = df.groupby('belongs_to_collection.id')['belongs_to_collection.id']\
                                    .transform('count').fillna(0).astype(int).copy()

    # Company with most productions: (In data)
    company_size_dict = get_element_frequency(df, 'production_companies.id') # {company_id : company_size}
    df['biggest_production_company_size'] = df['production_companies.id']\
                                        .apply(lambda companies: map_and_max(companies, company_size_dict))\
                                        .fillna(0).astype(int)

    # Country with most production companies
    id_country_set = set(df.apply(lambda x: tuple(zip(x['production_companies.id'], x['production_companies.origin_country'])) 
                                            if x['production_companies.origin_country'] else tuple(), 
                                    axis=1).sum())
    company_per_country = Counter(country for comp_id, country in id_country_set)
    company_per_country[''] = 0 # Update no-countries to 0
    df['most_companies_country_size'] = df['production_companies.origin_country']\
                                    .apply(lambda companies: map_and_max(companies, company_per_country))\
                                    .fillna(0).astype(int)
    
    # Largest production country size:
    country_size_dict = get_element_frequency(df, 'production_countries') # {country : movie_count}
    df['most_productions_country_size'] = df['production_countries']\
                                        .apply(lambda countries: map_and_max(countries, country_size_dict))\
                                        .fillna(0).astype(int)
    # Males/ Females+Males ratio:
    df['cast.gender_ratio'] = df['cast.gender'].apply(genders_ratio)

    # Num of spoken languages
    df['spoken_lang_num'] = df.spoken_languages.apply(len)

    # Word\Char count:
    df['overview_word_count'] = df.overview.apply(lambda x: smart_len(x, ' ')) # Overview word-count
    df['tagline_char_count'] = df.tagline.apply(smart_len) # tagline character-count
    df['title_char_count'] = df.title.apply(smart_len) # title character-count

    # Cast size:
    df['cast_size'] = df['cast.id'].apply(smart_len)

    # Crew size:
    df['crew_size'] = df['crew.id'].apply(smart_len)

    # Dept. size:
    dept_size_df = df['crew.department'].apply(lambda x: pd.Series(Counter(x)))\
                        .add_suffix('_depart_size')\
                        .astype('Int64')
    dept_size_df.dropna(axis=1, thresh= dept_size_df.shape[0] * 0.20, inplace=True) # Drop columns with less than 20% data
    dept_size_df.fillna(0, inplace=True) # Missing value imputation with 0
    df = pd.concat([df, dept_size_df[sorted(dept_size_df.columns)]], axis=1)

    # Mean by years:
    mean_by_year = df.groupby("release_year")[['runtime', 'budget', 'popularity']]\
                        .aggregate('mean')\
                        .rename(columns= {  'runtime' : 'avg_runtime_by_year',
                                            'budget' : 'avg_budget_by_year',
                                            'popularity' : 'avg_popularity_by_year'})
    df = df.join(mean_by_year, how='left', on='release_year')

    # Original title changed:
    df['title_changed'] = (df['original_title'] != df['title'])

    return df

def feature_engineering(df: pd.DataFrame, test: bool):

    # Drop unnecessary features:
    X = drop_features(df)

    Y = X['revenue'].copy()
    X.drop('revenue', axis=1, inplace=True)

    # Flatten and extract:
    X = features_flattening(X)
    
    # Extract those features!
    X = feature_extraction(X)

    # Add dummies:
    if test:
        X = add_dummies_test(X)
    else:
        X = add_dummies_train(X)

    # Drop unneccesry fields:
    tuple_fields = ['genres', 'spoken_languages', 'production_countries', 'production_companies.id', 'Keywords.id', 'cast.id', 'cast.gender', 'crew.id', 'crew.department', 'belongs_to_collection.id', 'production_companies.origin_country']
    text_fields = ['original_language', 'original_title', 'overview', 'tagline', 'title']

    X.drop(tuple_fields+text_fields, axis=1, inplace=True)

    # Imputate missing values of budget and runtime:
    X = missing_value_imputation(X)

    # Log scale big numbers:
    X['budget'] = X.budget.transform(np.log1p)

    return X, Y
