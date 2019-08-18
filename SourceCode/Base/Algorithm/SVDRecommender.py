"""
@author: Lim Yuan Her
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import os
from Base.Algorithm.AlgoBase import AlgoBase


class SVDRecommender(AlgoBase):

    def __init__(self, dsReader, fitModeAtInitialize=True):
        AlgoBase.__init__(self, dsReader, fitModeAtInitialize)
        self.RECOMMENDER_NAME = "SVD Recommender"
        
        foldername, filename = "Models", "SVDModel.pkl"

        if(self.fitModeAtInitialize == True):
            if(AlgoBase.checkPathExists(self, foldername, filename) == True):
                self.preds_df = AlgoBase.loadModel(self, foldername, filename)
            else:
                self.fit()

    def fit(self):  
        
        U, s, Vt = svds(self.train, k = 100)

        sigma = np.diag(s)

        # Check number of terms of s to include to capture at least 90% of the information
        sum = 0
        tot = s.sum()
        ctr = 0
        while(sum < 0.9):
            sum += s[ctr]/tot
            ctr += 1
        print("K: {}, Cumulative Variance: {}".format(ctr, sum))

        self.all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        self.preds_df = pd.DataFrame(self.all_user_predicted_ratings, columns = self.URM.columns, index= self.URM.index)

        AlgoBase.saveModel(self, self.preds_df, "Models", "SVDModel.pkl")

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
            df_user_unreadbooks_pred_sorted = df_user_unreadbooks_pred.rename(columns = {user_id: 'Predictions'}).sort_values('Predictions', ascending = False)

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