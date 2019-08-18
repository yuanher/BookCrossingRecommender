"""
@author: Lim Yuan Her
"""
import os

import logging
logging.getLogger('tensorflow').disabled = True

import numpy as np
import pandas as pd

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.utils import shuffle
from sklearn import preprocessing

from Base.Algorithm.AlgoBase import AlgoBase


class EmbeddingsRecommender(AlgoBase):

    RECOMMENDER_NAME = "Embeddings Recommender"

    def __init__(self, dsReader, fitModeAtInitialize=True):
        AlgoBase.__init__(self, dsReader, fitModeAtInitialize)
        self.RECOMMENDER_NAME = "Embeddings Recommender"

        # Define N and P - number of users and books respectively
        self.N = len(self.RatingsM["User-ID"].unique().tolist())
        self.P = len(self.RatingsM["ISBN"].unique().tolist())
                
        foldername, filename = "Models", "EMBModel.h5"
        self.filepath = os.path.dirname(os.path.abspath(__file__))
        self.filepath = self.filepath + "/" + foldername + "/" + filename

        if(self.fitModeAtInitialize == True):
            if(AlgoBase.checkPathExists(self, foldername, filename) == True):
                #self.model = AlgoBase.loadModel(self, foldername, filename)
                self.model = tf.keras.models.load_model(self.filepath)
            else:
                self.fit()

    def fit(self):  

        # Label encode user-IDs and ISBNs to within maximum number of users and books respectively
        le = preprocessing.LabelEncoder()

        self.RatingsM['ISBN_enc'] = le.fit_transform(self.RatingsM['ISBN'])
        self.RatingsM['User-ID_enc'] = le.fit_transform(self.RatingsM['User-ID'])

        # Define K embedding dimensionality, and epochs
        K = 100 # latent dimensionality
        epochs = 2

        # Define Inputs and Embeddings for the Keras model
        u = Input(shape=(1,))
        p = Input(shape=(1,))
        u_embedding = Embedding(self.N, K)(u)
        p_embedding = Embedding(self.P, K)(p)

        # Define x as the output for the Keras model
        x = Dot(axes=2)([u_embedding, p_embedding])
        x = Flatten()(x)

        # Define the Keras model
        self.model = Model(inputs = [u,p], outputs = x)

        self.model.compile (
            loss='mse',
            optimizer=SGD(lr=0.08, momentum=0.9),
            metrics = ['mse']
        )

        # Train the Keras model with RatingsM data
        r = self.model.fit(
            x = [self.RatingsM["User-ID_enc"].tolist(), self.RatingsM["ISBN_enc"].tolist()],
            y = self.RatingsM["Book-Rating"].tolist(),
            epochs = epochs,
            batch_size = 128,
            validation_data=(
                [self.RatingsM["User-ID_enc"].tolist(), self.RatingsM["ISBN_enc"].tolist()],
                self.RatingsM["Book-Rating"].tolist()
            ),
            verbose=0
        )

        self.model.save(self.filepath)

    def recommend(self, user_id, K=5):    
        try:
            mean_rating = self.dsReader.getAvgRatingByUser(user_id)

            arr = np.zeros((len(self.URM.index), len(self.URM.columns)))
            self.preds_df = pd.DataFrame(arr, columns = self.URM.columns, index= self.URM.index)

            for j in range(self.P):
                book_isbn = self.URM.columns[j]
                pred_rating = self.model.predict([np.array([0]), np.array([j])])

                self.preds_df.loc[user_id, book_isbn] = pred_rating + mean_rating

            # Get and sort the user's predictions
            sorted_user_predictions = self.preds_df.loc[user_id].sort_values(ascending=False)
            
            # Generates list of books read and rated by user
            df_user_book_list = self.dsReader.getUserBookList(user_id)
            df_user_book_list = df_user_book_list[["ISBN", "Book-Title", "Book-Author", "Book-Rating"]]

            # Get books that user has not read yet
            df_user_unreadbooks = self.BooksM[~self.BooksM['ISBN'].isin(df_user_book_list['ISBN'])]
            
            # Combine books that user has not read yet with predicted rating
            df_user_unreadbooks_pred = df_user_unreadbooks.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'ISBN', right_on = 'ISBN')
            
            # Sort result based on predicted rating column in descending order
            df_user_unreadbooks_pred_sorted = df_user_unreadbooks_pred.rename(columns = {user_id: 'Predictions'}).sort_values('Predictions', ascending = False)

            # Remove duplicate records of books with same name
            df_PredictList_all = df_user_unreadbooks_pred_sorted.drop_duplicates(subset='Book-Title', keep='first')

            # Remove records with no matching details from Books dataset
            df_PredictList_all = df_PredictList_all.dropna(subset=["Book-Title"])
                    
            # Get top K recommended books
            df_PredictList = df_PredictList_all.iloc[:K, :]

            df_PredictList = df_PredictList[["ISBN", "Book-Title", "Book-Author"]]

        except Exception as e:
            print(e)
            df_PredictList = None
            df_user_book_list = None

        return df_PredictList, df_user_book_list

    def compareActualPredictedRatings(self, user_id, userRecBookList):
        userRecBookList_ISBNs = list(userRecBookList.keys())

        # Get and sort the user's predictions
        sorted_user_predictions = self.preds_df.loc[user_id].sort_values(ascending=False)
        predictedRatings = pd.DataFrame(sorted_user_predictions).reset_index()

        # Get predicted ratings for user recommended booklist
        predictedRatings = predictedRatings[predictedRatings["ISBN"].isin(userRecBookList_ISBNs)].rename(columns = {user_id: 'Predicted-Rating'})

        # Get actual ratings for user recommended booklist
        actualRatings = self.dsReader.getRatingsByISBNs(user_id, userRecBookList_ISBNs)

        # Merge both predicted and actual ratings
        allRatings = actualRatings.merge(predictedRatings, how = 'left', left_on = 'ISBN', right_on = 'ISBN')

        # Get book details based on ISBN 
        allRatings = allRatings.merge(self.BooksM[['ISBN', 'Book-Title', 'Book-Author']], how = 'left', left_on = 'ISBN', right_on = 'ISBN')
        allRatings = allRatings[['ISBN', 'Book-Title', 'Book-Author', 'Predicted-Rating', 'Book-Rating']]

        # Evaluate RMSE metric for predicted/actual ratings
        predicted = np.asarray(allRatings["Predicted-Rating"].tolist())
        actual = np.asarray(allRatings["Book-Rating"].tolist())
        eval_res = AlgoBase.Evaluate(self, predicted, actual, ["rmse"])

        return allRatings, eval_res