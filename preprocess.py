
import numpy 
import pickle
import pandas as pd
game_name = '0021500009.pkl'

game = pd.read_pickle('/Users/ambroseling/Desktop/raptors/data/'+game_name)
print(game.keys())
print(game['events'][0].keys())
print(game['events'][0]['moments'][0].keys())

