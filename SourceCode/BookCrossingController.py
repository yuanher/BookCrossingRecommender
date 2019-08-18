from Base.Algorithm.CFRecommender import CFRecommender
from Base.Algorithm.TfidfRecommender import TfidfRecommender
from Base.Algorithm.SVDRecommender import SVDRecommender
from Base.Algorithm.SGDRecommender import SGDRecommender
from Base.Algorithm.EmbeddingsRecommender import EmbeddingsRecommender
from Base.Algorithm.AlgoBase import AlgoBase
from BookCrossingView import BookCrossingView, VIEW_OPTS, RECOMMENDER_TYPE
from Base.DataLoader.DatasetReader import DatasetReader

class BookCrossingController(object):

    def __init__(self, view):
        print("Loading Data, please wait...")        
        self.dsReader = DatasetReader()    
        self.dsReader.loadData()
        self.view = view
        self.selectedOption = 0

    def showMenu(self):
        while(self.selectedOption != VIEW_OPTS.QUIT.value):
            self.selectedOption = self.view.showMenu()
            if(self.selectedOption == VIEW_OPTS.ADD_USER.value):
                location, age = self.view.getNewUserDetails()
                self.userId = self.dsReader.addUser(location, age)
                self.view.displayUserDetails(self.userId, location, age, defaultPrompt=True)
            elif(self.selectedOption == VIEW_OPTS.ADD_RATING.value):
                while True: 
                    self.view.showHeaderMsg("Add Ratings for User")
                    self.userId = self.view.getUserID(1, 1000000, cancel=True)
                    try:
                        self.validUserIDs = self.dsReader.getUserIDs("User")
                        invalidUserID = self.userId not in self.validUserIDs
                        if(invalidUserID == True):
                            raise Exception('')
                    except Exception as e:
                        self.view.showErrorMsg("User not found. Please try again.")
                        continue
                    else:
                        break

                if(not invalidUserID):
                    ratings = self.view.getNewUserRatingDetails()
                    self.dsReader.addRatings(self.userId, ratings)
                    self.update_models(RECOMMENDER_TYPE.ALL.value)
                    
            elif(self.selectedOption == VIEW_OPTS.GET_RECS.value):
                while True: 
                    self.view.showHeaderMsg("Get Recommendations for user")
                    self.userId = self.view.getUserID(1, 1000000, cancel=True)
                    try:
                        self.validUserIDs = self.dsReader.getUserIDs("Ratings")
                        invalidUserID = self.userId not in self.validUserIDs
                        if(invalidUserID == True):
                            raise Exception('')
                    except Exception as e:
                        self.view.showErrorMsg("No ratings for user {} found. Please add ratings for this user first.".format(self.userId))
                        continue
                    else:
                        break

                if(not invalidUserID):

                    self.recType = self.view.getUserRecsDetails()

                    if(self.recType != 99):
                        # Get recommender object based on selected type
                        self.recommender_object = self.__getRecommender(self.userId, self.recType)

                        # Get list of user recommended booklist and rated list
                        df_recList, df_userBookList = self.recommender_object.recommend(self.userId, 5)
                        if(df_recList is None and df_userBookList is None):
                            self.view.showErrorMsg("No recommendations for {} available".format(self.userId))
                        else:
                            rateRecItems = self.view.displayRecs(df_recList, df_userBookList)

                            if(rateRecItems == 1):
                                # Get dictionary object of user recommended booklist
                                recBookList = df_recList[['ISBN', 'Book-Title']].set_index('ISBN').to_dict()['Book-Title']

                                # Get user rating for recommended booklist
                                ratings = self.view.getUserRecRatingDetails(recBookList)
                                for key, value in ratings.items():
                                    self.dsReader.addRating(self.userId, key, ratings[key])

                                # Get predicted and user ratings for recommended booklist 
                                df_rec_rated_List, eval_res = self.recommender_object.compareActualPredictedRatings(self.userId, recBookList)

                                # Display predicted and user ratings for recommended booklist
                                self.view.displayRecsWithRatings(df_rec_rated_List, eval_res)
                            else:
                                self.view.showInfoMsg("Recommended book list rating cancelled", "Return to main menu", True)
            elif(self.selectedOption == VIEW_OPTS.QUIT.value):
                self.view.quit()
            elif(self.selectedOption == VIEW_OPTS.UPDATE_MDL.value):
                self.recType = self.view.updateModel()
                if(self.recType != 99):
                    self.update_models(self.recType)
        
    def __getRecommender(self, userId, recType):
        try:
            userLocation, userAge, userNumRating = self.dsReader.getUserDetails(userId)
            self.view.displayUserStats(userId, userLocation, userAge, userNumRating)

            if(recType == RECOMMENDER_TYPE.TFIDF.value):
                rec = TfidfRecommender(self.dsReader, fitModeAtInitialize=True)
            elif(recType == RECOMMENDER_TYPE.USERCF.value):
                rec = CFRecommender(self.dsReader, fitModeAtInitialize=True)
                rec.setCFtype("user")
            elif(recType == RECOMMENDER_TYPE.ITEMCF.value):
                rec = CFRecommender(self.dsReader, fitModeAtInitialize=True)
                rec.setCFtype("item")
            elif(recType == RECOMMENDER_TYPE.SVD.value):  
                rec = SVDRecommender(self.dsReader, fitModeAtInitialize=True)        
            elif(recType == RECOMMENDER_TYPE.SGD.value):  
                rec = SGDRecommender(self.dsReader, fitModeAtInitialize=True)       
            elif(recType == RECOMMENDER_TYPE.EMB.value):  
                rec = EmbeddingsRecommender(self.dsReader, fitModeAtInitialize=True) 

            return rec
        except Exception as e:
            print(e)

    def update_models(self, recType):
        if(recType == RECOMMENDER_TYPE.TFIDF.value):
            TfidfRecommender(self.dsReader, fitModeAtInitialize=False).fit()
        elif(recType == RECOMMENDER_TYPE.USERCF.value):
            CFRecommender(self.dsReader, fitModeAtInitialize=False).fit(similarity = "cosine", CFtype="user")
        elif(recType == RECOMMENDER_TYPE.ITEMCF.value):
            CFRecommender(self.dsReader, fitModeAtInitialize=False).fit(similarity = "cosine", CFtype="item")
        elif(recType == RECOMMENDER_TYPE.SVD.value):
            SVDRecommender(self.dsReader, fitModeAtInitialize=False).fit()            
        elif(recType == RECOMMENDER_TYPE.SGD.value):
            SGDRecommender(self.dsReader, fitModeAtInitialize=False).fit()
        elif(recType == RECOMMENDER_TYPE.EMB.value):
            EmbeddingsRecommender(self.dsReader, fitModeAtInitialize=False).fit()
        elif(recType == RECOMMENDER_TYPE.ALL.value):
            title = "Updating recommender models...\n"
            self.view.showInfoMsgWithProgressBar(title, "Updating TF-IDF Recommender model...\n", 0)
            TfidfRecommender(self.dsReader, fitModeAtInitialize=False).fit()
            self.view.showInfoMsgWithProgressBar(title, "TF-IDF Recommender model update completed.\nUpdating User Based CF Recommender model...\n", 1)
            CFRecommender(self.dsReader, fitModeAtInitialize=False).fit(similarity = "cosine", CFtype="user")
            self.view.showInfoMsgWithProgressBar(title, "User Based CF Recommender model update completed.\nUpdating Item Based CF Recommender model...\n", 2)
            CFRecommender(self.dsReader, fitModeAtInitialize=False).fit(similarity = "cosine", CFtype="item")
            self.view.showInfoMsgWithProgressBar(title, "Item Based CF Recommender model update completed.\nUpdating SVD Recommender model...\n", 3)
            SVDRecommender(self.dsReader, fitModeAtInitialize=False).fit() 
            self.view.showInfoMsgWithProgressBar(title, "SVD Recommender model update completed.\nUpdating SGD Recommender model...\n", 4)
            SGDRecommender(self.dsReader, fitModeAtInitialize=False).fit()
            self.view.showInfoMsgWithProgressBar(title, "SGD Recommender model update completed.\nUpdating Embeddings Recommender model...\n", 5)            
            EmbeddingsRecommender(self.dsReader, fitModeAtInitialize=False).fit()
            self.view.showInfoMsgWithProgressBar(title, "Embeddings Recommender model update completed.\n", 6, True)  