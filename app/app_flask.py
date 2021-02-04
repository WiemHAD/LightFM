from flask import Flask, render_template, request
from fonctions import data, get_recommendations, new_user, rating
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix

app = Flask(__name__)

ap=data()
ratings,artist_names, n_items = rating(ap)
@app.route('/')
def hello():
    artist_name = ap.name

    return render_template('index.html',artist_name= artist_name)


@app.route('/page2', methods = ['POST'])
def page2():
    selected_Artist = request.form.getlist('selectedArtist')
    print(selected_Artist)
    ratings_df = ap.pivot(index='userID', columns='artistID', values='playCountScaled')
    ratings = ratings_df.fillna(0).values
    nouveau_user = new_user(selected_Artist,ratings,artist_names)
    ratings = np.vstack((ratings,nouveau_user))
    X = csr_matrix(ratings)
    reco = get_recommendations(ratings.shape[0]-1,artist_names,n_items,X)
 
    return render_template("page2.html", liste_Artistes = selected_Artist,reco=reco)




if __name__ == "__main__":
    app.run()

