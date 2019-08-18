"""
@author: Lim Yuan Her
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from Base.Algorithm.AlgoBase import AlgoBase

class TfidfRecommender(AlgoBase):

    def __init__(self, dsReader, fitModeAtInitialize=True):
        AlgoBase.__init__(self, dsReader, fitModeAtInitialize)
        self.RECOMMENDER_NAME = "TF=IDF Recommender"

        foldername, filename = "Models", "TfidfModel.pkl"

        if(self.fitModeAtInitialize == True):
            if(AlgoBase.checkPathExists(self, foldername, filename) == True):
                self.similarityM = AlgoBase.loadModel(self, foldername, filename)
            else:
                self.fit()

    def fit(self, similarity="cosine"):     
        try:
            corpus = self.BooksM[self.BooksM.ISBN.isin(self.RatingsM.ISBN)]["Book-Title"].tolist()
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(corpus)
            self.X_df = pd.DataFrame(X.todense(), columns = vectorizer.get_feature_names(), index=corpus)
            self.Similarity_Matrix = self.Compute_Similarity(similarity)
            AlgoBase.saveModel(self, self.Similarity_Matrix, "Models", "TfidfModel.pkl")
        except Exception as e:
            print(e)

    def Compute_Similarity(self, similarity):
        self.similarity = similarity
        title_labels = self.X_df.index
        #id_labels = [self.BooksM[self.BooksM['Book-Title'] == label]['ISBN'].values[0] for label in title_labels]

        if(similarity == "corr"):
            #self.similarityM = self.X_df.corr().round(4)
            sim_array = np.corrcoef(self.X_df.values)
            self.similarityM = pd.DataFrame(data=sim_array, index=title_labels, columns=title_labels)
        elif(similarity == "cosine"):
            sim_array = cosine_similarity(self.X_df)
            self.similarityM = pd.DataFrame(data=sim_array, index=title_labels, columns=title_labels)
        elif(similarity == "adj_cosine"):
            M = np.asarray(self.X_df)
            M_mean = M.mean(axis=1)
            M_mean_adj = M - M_mean[:, None]
            sim_array = cosine_similarity(M_mean_adj)
            self.similarityM = pd.DataFrame(data=sim_array, index=title_labels, columns=title_labels)
        elif(similarity == "euclidean"):
            sim_array = euclidean_distances(self.X_df)
            self.similarityM = pd.DataFrame(data=sim_array, index=title_labels, columns=title_labels)

        return self.similarityM

    def recommend(self, user_id, K = 5):
        try:
            # Generates list of books read and rated by user
            df_user_book_list = self.dsReader.getUserBookList(user_id)

            # Gets list of books rated higher than 5.0 by user
            df_user_book_list_HR = df_user_book_list[df_user_book_list["Book-Rating"] >= 5.0].sort_values(by='Book-Rating', ascending=False)

            df_user_book_list_HR = df_user_book_list_HR[["ISBN", "Book-Title", "Book-Author", "Book-Rating"]]

            # Combine similarity ratings for high-rated books
            df_similarityM = self.similarityM.reset_index().rename(columns={self.similarityM.index.name:'Book-Title'})
            
            df_user_book_list_HR_sim = df_user_book_list_HR.merge(df_similarityM, how = 'left', left_on = 'Book-Title', right_on = 'index')
            
            # Get most similar book for each high-rated book
            if(self.similarity != "euclidean"):
                sort_asc = False
            else:
                sort_asc = True

            df_user_book_list_HR_sim = df_user_book_list_HR_sim[self.similarityM.columns.append(pd.Index(["Book-Title"]))].set_index('Book-Title').apply(lambda x: pd.Series(x.sort_values(ascending=sort_asc).iloc[:2].index, index=['top1','top2']), axis=1).reset_index()

            #remove book of interest from similar book list
            def remove_self (row):
                if row['top1'] == row['Book-Title']:
                    return row['top2']
                elif row['top2'] == row['Book-Title']:
                    return row['top1']
                else:
                    return row['top1']

            df_user_book_list_HR_sim['Max'] = df_user_book_list_HR_sim.apply(lambda row: remove_self(row), axis=1)
            
            df_PredictList_all = pd.DataFrame(pd.DataFrame({'Book-Title':df_user_book_list_HR_sim.Max.tolist()}))

            df_PredictList_all = df_PredictList_all[~df_PredictList_all["Book-Title"].isin(df_user_book_list_HR["Book-Title"])]

            # Order list by number and average ratings
            d = {'Book-Title': ['count']}
            res = df_PredictList_all.groupby('Book-Title').agg(d)
            res.columns = res.columns.droplevel(0)
            res = res.sort_values(by=["count"], ascending=False)

            df_PredictList_all = res.merge(self.BooksM, how = 'left', left_on = 'Book-Title', right_on = 'Book-Title')      
        
            # Remove duplicate records of books with same name
            df_PredictList_all = df_PredictList_all.drop_duplicates(subset='Book-Title', keep='first')

            # Remove records with no matching details from Books dataset
            df_PredictList_all = df_PredictList_all.dropna(subset=["Book-Title"])

            # Get top K recommended books
            df_PredictList = df_PredictList_all.iloc[:K, :]

            df_PredictList = df_PredictList[["ISBN", "Book-Title", "Book-Author"]] 

        except Exception as e:
            df_PredictList = None
            df_user_book_list_HR = None

        return df_PredictList, df_user_book_list_HR

    def compareActualPredictedRatings(self, user_id, userRecBookList):
        userRecBookList_ISBNs = list(userRecBookList.keys())

        # Get actual ratings for user recommended booklist
        actualRatings = self.dsReader.getRatingsByISBNs(user_id, userRecBookList_ISBNs)

        # Get book details based on ISBN 
        allRatings = actualRatings.merge(self.BooksM[['ISBN', 'Book-Title', 'Book-Author']], how = 'left', left_on = 'ISBN', right_on = 'ISBN')
        allRatings = allRatings[['ISBN', 'Book-Title', 'Book-Author', 'Book-Rating']]

        return allRatings, None