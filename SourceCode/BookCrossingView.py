# importing enum for enumerations 
import enum 
import os
import sys
import pandas as pd

# creating enumerations using class 
class VIEW_OPTS(enum.Enum): 
    ADD_USER = 1
    ADD_RATING = 2
    GET_RECS = 3
    UPDATE_MDL = 4
    QUIT = 5

class RECOMMENDER_TYPE(enum.Enum): 
    TFIDF = 1
    USERCF = 2
    ITEMCF = 3
    SVD = 4
    SGD = 5
    EMB = 6
    ALL = 7

class BookCrossingView(object):

    def getValidNumEntry(self, user_prompt, minVal, maxVal, cancel=False):
        while True: 
            user_in = input(user_prompt)
            try:
                result = int(user_in)
                if cancel == True:
                    InvalidCond = (result < minVal or result > maxVal) and result != 99
                else:
                    InvalidCond = (result < minVal or result > maxVal)
                if(InvalidCond == True):
                    raise Exception('')
            except Exception as e:
                print("Invalid entry. Please enter value between {} and {}".format(minVal, maxVal))
                continue
            else:
                break
        return result
        
    def showMenu(self):

        os.system('cls||clear')
        print("********************************")
        print("Book Crossing Recommender System")
        print("********************************")
        print("1. Add User")   
        print("2. Add Rating") 
        print("3. Get recommendation for user") 
        print("4. Update Model") 
        print("5. Quit") 
        print()     

        user_prompt = "Select an option: "
        option = self.getValidNumEntry(user_prompt, 1, 5, cancel=False)

        return option

    def getNewUserDetails(self):
        os.system('cls||clear')
        print("************")
        print("Add New User")
        print("************")
        print()

        location = input("Enter user location: ")
        print()
        age = self.getValidNumEntry("Enter user age: ", 1, 125, cancel=False)

        return location, age

    def getUserID(self, minVal, maxVal, cancel=False):
        userId = self.getValidNumEntry("Enter user ID: ", minVal, maxVal, cancel)
        print()        
        return userId

    def getNewUserRatingDetails(self):
        option = 0
        ratings = {}
        book_isbns = {"0385504209":"The Da Vinci Code", 
                      "044021145X": "The Firm",
                      "0440214041": "The Pelican Brief",
                      "0345337662": "Interview with the Vampire",
                      "059035342X": "Harry Potter and the Sorcerer's Stone",
                      "0345370775": "Jurassic Park",
                      "0156027321": "Life of Pi",
                      "0446310786": "To Kill a Mockingbird",
                      "0440206154": "Red Dragon",
                      "0345342968": "Fahrenheit 451",
                      "0399501487": "Lord of the Flies",
                      "0312924585": "Silence of the Lambs",
                      "0345339681": "The Hobbit",
                      "0679745203": "The English Patient",
                      "0671041789": "The Green Mile"
        }

        print("Please rate the following items (1-10): ")
        for key, value in book_isbns.items():
            print()
            user_prompt = book_isbns[key] + " (" + str(key) + "): "
            option = self.getValidNumEntry(user_prompt, 1, 10, cancel=False)
            ratings[key] = option
            option = input("Press 'q' to finish, other key to continue: ")
            if(option == 'q'):
                break

        return ratings

    def getUserRecRatingDetails(self, recBookList):
        option = 0
        ratings = {}
        books_rated = {}       

        os.system('cls||clear')
        print("Rate recommended books:")
        print("***********************")
        print()
        print("Please rate on a scale of (1-10): ")
        print()
        for key, value in recBookList.items():
            user_prompt = recBookList[key] + " (" + str(key) + "): "
            option = self.getValidNumEntry(user_prompt, 1, 10, cancel=False)
            ratings[key] = option
            #ratings.append(option)
        print()
        print("Updating model and evaluating results, please wait...")
        print()

        return ratings

    def getUserRecsDetails(self):
        recType = self.getRecType(disableAll = True)

        return recType

    def updateModel(self):
        os.system('cls||clear')
        print("************")
        print("Update Model")
        print("************")

        recType = self.getRecType(disableAll = False)

        return recType

    def getRecType(self, disableAll = True):
        print() 
        print("1. Content-Based")   
        print("2. User-based Collaborative-Filtering")  
        print("3. Item-based Collaborative-Filtering")  
        print("4. SVD")  
        print("5. SGD")
        print("6. Embeddings")  
        if not disableAll:
            print("7. All (Warning: Update of all models will take a long time)")
        print("99. Cancel")
        print()     
        recType = self.getValidNumEntry("Select Recommender Type: ", 1, 7, cancel=True)

        return recType

    def displayUserStats(self, id, location, age, numratings):
        os.system('cls||clear')
        print("User Details:")
        print("*************")
        print()
        print("ID:                {}".format(id))        
        print("Location:          {}".format(location))        
        print("Age:               {}".format(age))
        print("Number of Ratings: {}".format(numratings))
        print()

    def displayRecs(self, df_recList, df_bookList):
        os.system('cls||clear')
        print("User Rated Book List:")
        print("*********************")
        print()
        print(df_bookList.head(5))
        print()
        key = input("Press any key to see recommended book list...")
        os.system('cls||clear')
        print("User Recommended Book List:")
        print("***************************")
        print()
        print(df_recList)
        print()
        key = self.getValidNumEntry("Rate recommended book list? (1 - Yes, 2 - No)", 1, 2, cancel=False)
        return key

    def displayRecsWithRatings(self, df_recList, eval_res):
        os.system('cls||clear')
        if(eval_res is not None):
            print()
            print("Evaluation Results:")
            for key, value in eval_res.items():
                print("{}: {:.2f}".format(key, value))
        print()
        print("User Rated Recommended Book List:")
        print("*********************************")
        print()
        print(df_recList.head(5))
        print()
        key = input("Press any key to return to main Menu...")
        
    def displayUserDetails(self, userId, location, age, defaultPrompt=False):
        print()
        print("New User ID: {}".format(userId))
        print()
        if(defaultPrompt == True):
            key = input("Press any key to continue...")
        else:
            key = input("Press any key to see user rated book list...")

    def quit(self):
        os.system('cls||clear')
        print("Exiting Program...")
        exit()

    def showErrorMsg(self, errMsg):
        print()
        print(errMsg)
        print()  
        key = input("Press any key to continue...")
    
    def showInfoMsgWithProgressBar(self, Title, Msg, i, pauseMode=False):
        os.system('cls||clear')
        print(Title)
        print()
        print(Msg)
        self.loadingBar(i, 6, 2)
        if(pauseMode):
            key = input("Press any key to continue...")

    def showInfoMsg(self, Title, Msg, pauseMode=False):
        os.system('cls||clear')
        print(Title)
        print()
        print(Msg)
        print()
        if(pauseMode):
            key = input("Press any key to continue...")

    def showHeaderMsg(self, Msg):
        i = len(Msg)
        os.system('cls||clear')
        print("*" * i)
        print(Msg)
        print("*" * i)
        print()

    def loadingBar(self, count,total,size):
        percent = (float(count) / float(total)) * 100
        print("\r" + str(int(count)).rjust(3,'0')+"/"+str(int(total)).rjust(3,'0') + ' [' + '='*int(percent/10)*size + ' '*(10-int(percent/10))*size + ']')