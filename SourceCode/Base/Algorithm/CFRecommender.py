"""
@author: Lim Yuan Her
"""

import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import mean_squared_error
import os
import sys
from Base.Algorithm.AlgoBase import AlgoBase

class CFRecommender(AlgoBase):

    def __init__(self, dsReader, fitModeAtInitialize=True):
        AlgoBase.__init__(self, dsReader, fitModeAtInitialize)
        self.RECOMMENDER_NAME = "User-Item Based Collaborative Filtering Recommender"

    def setCFtype(self, cftype):
        self.CFtype = cftype
        foldername, filename = "Models", self.CFtype + "_CFModel.pkl"

        if(self.fitModeAtInitialize == True):
            if(AlgoBase.checkPathExists(self, foldername, filename) == True):
                self.preds_df = AlgoBase.loadModel(self, foldername, filename)
            else:
                self.fit(similarity = "cosine", CFtype="user")
                self.fit(similarity = "cosine", CFtype="item")

    def fit(self, similarity = "cosine", CFtype="user"):  

        self.similarity = similarity
        self.CFtype = CFtype
        self.Similarity_Matrix = self.Compute_Similarity(self.similarity, self.train, self.CFtype)
        self.user_prediction = self.predict(self.train, self.Similarity_Matrix, self.CFtype) 
        self.preds_df = pd.DataFrame(self.user_prediction, columns = self.URM.columns, index= self.URM.index)
        AlgoBase.saveModel(self, self.preds_df, "Models", self.CFtype + "_CFModel.pkl")

    def Compute_Similarity(self, similarity, ratings, CFtype):
        if CFtype == "user":
            ratings = ratings
        elif CFtype == "item":
            ratings = ratings.T

        if(similarity == "corr"):
            sim_array_df = pd.DataFrame(data=ratings.T)
            sim_array = sim_array_df.corr(method='pearson', min_periods=100).round(4)
        elif(similarity == "cosine"):
            sim_array = cosine_similarity(ratings)
        elif(similarity == "adj_cosine"):
            M_mean = ratings.mean(axis=1)
            M_mean_adj = ratings - M_mean[:, None]
            sim_array = cosine_similarity(M_mean_adj)
        elif(similarity == "euclidean"):
            sim_array = euclidean_distances(ratings)
        return sim_array

    def recommend(self, user_id, K=5):
        try:
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
            df_user_unreadbooks_pred_sorted = df_user_unreadbooks_pred.rename(columns = {user_id: 'WeightedRank'}).sort_values('WeightedRank', ascending = False)

            # Remove duplicate records of books with same name
            df_PredictList_all = df_user_unreadbooks_pred_sorted.drop_duplicates(subset='Book-Title', keep='first')

            # Remove records with no matching details from Books dataset
            df_PredictList_all = df_PredictList_all.dropna(subset=["Book-Title"])
                    
            # Get top K recommended books
            df_PredictList = df_PredictList_all.iloc[:K, :]

            df_PredictList = df_PredictList[["ISBN", "Book-Title", "Book-Author"]]

        except Exception as e:
            df_PredictList = None
            df_user_book_list = None

        return df_PredictList, df_user_book_list

    # Function to predict ratings
    def predict(self, ratings, similarity, CFtype):
        mean_user_rating = np.array([ rating[np.nonzero(rating)[0]].mean() for rating in ratings])
        rating_bias = np.nan_to_num(mean_user_rating[:, np.newaxis], copy=False)
        rating_norm = np.array([np.abs(similarity).sum(axis=1)])

        if CFtype == 'user':
            simRatingProd = np.nan_to_num(similarity.dot(ratings), copy=False)
            pred = (simRatingProd / rating_norm.T) + rating_bias           
        elif CFtype == 'item':
            simRatingProd = np.nan_to_num(ratings.dot(similarity), copy=False)
            pred = (simRatingProd / rating_norm) + rating_bias

        return pred

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
            