"""
@author: Lim Yuan Her
"""

import numpy as np
import pandas as pd
import os
from Base.Algorithm.AlgoBase import AlgoBase


class SGDRecommender(AlgoBase):

    def __init__(self, dsReader, fitModeAtInitialize=True):
        AlgoBase.__init__(self, dsReader, fitModeAtInitialize)
        self.RECOMMENDER_NAME = "Stochastic Gradient Descent Recommender"
        
        foldername, filename = "Models", "SGDModel.pkl"
        if(self.fitModeAtInitialize == True):
            if(AlgoBase.checkPathExists(self, foldername, filename) == True):
                self.preds_df = AlgoBase.loadModel(self, foldername, filename)
            else:
                self.fit()

    def fit(self):  

        self.R = self.train
        self.num_users, self.num_items = self.R.shape
        self.K = 100
        self.alpha = 0.1
        self.beta = 0.01
        self.iterations = 50

        # Initialize user and item latent feature matrices
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            mse = self.mse()
            training_process.append((i, mse))
            if (i+1) % 1 == 0:
                print("Iteration = %d ; error = %.4f" % (i+1, mse))
        print("mse = %.4f" % (mse))

        self.preds_df = pd.DataFrame(data=self.full_matrix(), index=self.URM.index, columns=self.URM.columns)

        AlgoBase.saveModel(self, self.preds_df, "Models", "SGDModel.pkl")

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)


    def sgd(self):
        """
        Perform stochastic graident descent
        """
        import math
        for i, j, r in self.samples:
            # Computer prediction and error
            prediction = self.get_rating(i, j)
            e = (r - prediction)
            #print(e)
            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Update user and item latent feature matrices
            #self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            #self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

            P_i = np.copy(self.P[i,:])
            for k in range(self.K):
                self.P[i,k] += self.alpha * (e * self.Q[j,k] - self.beta * self.P[i,k])
                self.Q[j,k] += self.alpha * (e * P_i[k] - self.beta * self.Q[j,k])

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        #print(self.b, self.b_u[i], self.b_i[j])
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction


    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        self.sim_array = self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)
        return self.sim_array
        
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

