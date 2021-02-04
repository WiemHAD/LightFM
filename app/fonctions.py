import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from lightfm import LightFM

def data():
    plays = pd.read_csv('data/user_artists.dat', sep='\t')
    artists = pd.read_csv('data/artists.dat', sep='\t', usecols=['id','name'])

    # Merge artist and user pref data
    ap = pd.merge(artists, plays, how="inner", left_on="id", right_on="artistID")
    ap = ap.rename(columns={"weight": "playCount"})

    artist_rank = ap.groupby(['name']) \
    .agg({'userID' : 'count', 'playCount' : 'sum'}) \
    .rename(columns={"userID" : 'totalUsers', "playCount" : "totalPlays"}) \
    .sort_values(['totalPlays'], ascending=False)
    
    artist_rank['avgPlays'] = artist_rank['totalPlays'] / artist_rank['totalUsers']

    # Merge into ap matrix
    ap = ap.join(artist_rank, on="name", how="inner") \
    .sort_values(['playCount'], ascending=False)
    pc = ap.playCount
    play_count_scaled = (pc - pc.min()) / (pc.max() - pc.min())
    ap = ap.assign(playCountScaled=play_count_scaled)
    
    return ap

def rating(ap):
    # Build a user-artist rating matrix 
    ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
    ratings = ratings_df.fillna(0).values

    # Build a sparse matrix
    X = csr_matrix(ratings)

    n_users, n_items = ratings_df.shape
    user_ids = ratings_df.index.values
    artist_names = ap.sort_values("artistID")["name"].unique()
    #name_unique = list(set(ap['name']))
    return ratings, artist_names, n_items


def new_user(selectedArtist,ratings,artist_names):
    n_user = np.zeros(ratings.shape[1])
    artist_index=0
    for artist in artist_names:
        if artist in selectedArtist:
            n_user[artist_index] = np.mean(ratings[:,artist_index])
        artist_index+=1
    print(n_user)
    return n_user

def get_recommendations(user_id,artist_name,n_items,X):
      # initialize the model
    model = LightFM(learning_rate=0.05, loss='bpr', random_state=42)
    model.fit(X, epochs=10, num_threads=2)
    # predict
    scores = model.predict(user_id, np.arange(n_items))
    top_items = artist_name[np.argsort(-scores)]
    return(top_items[:10])



