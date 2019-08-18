"""
Created on 22/11/2018

@author: Lim Yuan Her
"""
from BookCrossingController import BookCrossingController
from BookCrossingView import BookCrossingView
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    BC_Ctr = BookCrossingController(BookCrossingView())
    BC_Ctr.showMenu()