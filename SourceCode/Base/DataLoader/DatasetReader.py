import os

import numpy as np
import pandas as pd

class DatasetReader:

    ratingsPath = 'DataSet/BX-Ratings.csv'
    booksPath = 'DataSet/BX-Books.csv'
    usersPath = 'DataSet/BX-Users.csv'

    def __init__(self):

        self.ratings = None
        self.books = None
        self.users = None
        self.ratings_matrix = None

    def loadData(self):

        # Load books data into memory
        df_books = pd.read_csv(self.booksPath, delimiter=",", error_bad_lines=False, encoding='latin-1', header=0, index_col=False)
        
        # Load users data into memory
        df_users = pd.read_csv(self.usersPath, delimiter=",", error_bad_lines=False, encoding='latin-1', header=0, index_col=False)

        # Load ratings data into memory
        df_ratings = pd.read_csv(self.ratingsPath, delimiter=",", header=0, encoding ='unicode_escape', index_col=False)

        self.ratings = df_ratings
        self.books = df_books[["ISBN", "Book-Title", "Book-Author",	"Year-Of-Publication",	"Publisher"]]
        self.users = df_users[["User-ID", "Location", "Age"]]

        self.ratings_matrix = df_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating')

        return (self.ratings_matrix, self.books, self.users, self.ratings)

    def getBookDetails(self, bookId):
        bookName = self.books[self.books["ISBN"] == bookId]["Book-Title"].values[0]
        bookAuthor = self.books[self.books["ISBN"] == bookId]["Book-Author"].values[0]
        bookYear = self.books[self.books["ISBN"] == bookId]["Year-Of-Publication"].values[0]
        return  (bookName, bookAuthor, bookYear)

    def getBookSubjects(self):
        # Get a series of strings from the Genres column of the dataframe
        s = self.books["Subjects"]

        def getSubjectList(subjectString):
            if str(subjectString) != "nan":
                d = subjectString.split(",")
            else:
                d = []
            return d

        # Create and populate dictionary containing all distinct subject
        subjectD = {}
                
        # The following prints out the number of disctinct subjects 
        for i in s:
            for j in getSubjectList(i):
                subjectD[j] = 0
        print ("Number of distinct subjects:", len(subjectD) )

        zeroArr = np.zeros(len(self.BooksM["Subjects"])).astype(int)

        for i in subjectD.keys():
            subjectD[i] = np.array(zeroArr)

        df2 = pd.DataFrame(subjectD)

        return df2

    def getRatingsMatrix(self):
        return self.ratings_matrix

    def getBooksData(self):
        return self.books

    def getUsersData(self):
        return self.users

    def getRatingsData(self):
        return self.ratings

    def getBookById(self, bookId):
        return self.books[self.books["ISBN"] == bookId]

    def getBooksByIds(self, book_ids):
        return self.books[self.books["ISBN"].isin(book_ids)][['ISBN', 'Book-Title', 'Book-Author',
                                                                'Year-Of-Publication', 'Publisher']]

    def getAvgRatingById(self, bookId):
        d = {'Book-Rating': ['mean']}
        res = self.ratings[self.ratings["ISBN"] == bookId].groupby('ISBN').agg(d)
        res.columns = res.columns.droplevel(0)
        res = res.rename({'mean': 'Book-Rating'}, axis=1)
        res = res.reset_index()

        return res

    def getAvgRatingByUser(self, userId):
        res = self.ratings[self.ratings["User-ID"] == userId]['Book-Rating'].mean()

        return res

    def getRatingsByISBNs(self, userId, ISBNs):
        cond1 = self.ratings["User-ID"] == userId
        cond2 = self.ratings["ISBN"].isin(ISBNs)
        return self.ratings[cond1 & cond2][["ISBN", "Book-Rating"]]

    def getUserReadBookNames(self, userId):
        userReadBookIds = self.ratings[self.ratings["User-ID"] == userId].ISBN.unique().tolist()
        return self.getBooksByIds(userReadBookIds)[["ISBN", "Book-Title"]]

    def getUserBookList(self, userId):
        # Get the user's data and merge in the book information.
        user_readbooks_df = self.ratings[self.ratings["User-ID"] == userId]
        user_book_list = user_readbooks_df.merge(self.books, how = 'left', left_on = 'ISBN', right_on = 'ISBN').sort_values(['Book-Rating'], ascending=False)

        return user_book_list

    def getUserIDs(self, ds):
        if(ds == "Ratings"):
            return self.ratings["User-ID"].unique().tolist()
        elif(ds == "User"):
            return self.users["User-ID"].unique().tolist()

    def getUserDetails(self, userId):
        location = self.users[self.users['User-ID'] == userId][['Location']].values[0][0]
        age = self.users[self.users['User-ID'] == userId][['Age']].values[0][0]
        numratings = self.ratings[self.ratings['User-ID'] == userId][['ISBN']].count()[0]

        return location, age, numratings

    def getBookIdByTitle(self, title):
        return self.books[self.books['Book-Title'] == title]['ISBN'].values[0]

    def getBookTitleByISBN(self, ISBN):
        self.books[self.books['ISBN'] == ISBN]['Book-Title'].values[0]

    def save_data(self, ds, ds_type):
        if ds_type == "user":
            ds.to_csv(self.usersPath, index=False)
        elif ds_type == "book":
            ds.to_csv(self.booksPath, index=False)    
        elif ds_type == "rating":   
            ds[['User-ID', 'ISBN', 'Book-Rating']].to_csv(self.ratingsPath, index=False)  

        self.loadData()

    def addUser(self, location, age):
        userId = self.users['User-ID'].max() + 1
        self.users = self.users.append({'User-ID':str(userId), 'Location':location, 'Age':age},                                                 ignore_index=True)
        self.save_data(self.users, "user")

        return userId

    def addBook(self, id, title, author, year, publisher, img_url_s, img_url_m="", img_url_l=""):
        self.books = self.books.append({'ISBN':id, 'Book-Title':title, 'Book-Author':author,
                           'Year-Of-Publication':year, 'Publisher':publisher, 'Image-URL-S':img_url_s, 
                           'Image-URL-M':img_url_m, 'Image-URL-L':img_url_l}, ignore_index=True)
        self.save_data(self.books, "book")
    
    def addRating(self, userId, id, rating):
        self.ratings = self.ratings.append({'User-ID':userId, 'ISBN':id, 'Book-Rating':rating},                                                     ignore_index=True)
        self.save_data(self.ratings, "rating") 

    def addRatings(self, userId, ratings):
        for key, value in ratings.items():
            self.ratings = self.ratings.append({'User-ID':userId, 'ISBN':key, 'Book-Rating':ratings[key]},                                            ignore_index=True)
        self.save_data(self.ratings, "rating")     

    def train_test_split(self):
        self.ratings_matrix = self.ratings_matrix.fillna(0)
        ratings = np.array(self.ratings_matrix)
        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in range(ratings.shape[0]):
            nonzero_ratings = ratings[user, :].nonzero()[0]
            if(len(nonzero_ratings) > 100):
                test_ratings = np.random.choice(nonzero_ratings, 
                                                size=10, 
                                                replace=False)
                train[user, test_ratings] = 0.0
                test[user, test_ratings] = ratings[user, test_ratings]

        return train, test
