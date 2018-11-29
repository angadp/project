'''
Created on Nov 25, 2018
@author: ishank
'''
import json
# from nltk.tokenize import RegexpTokenizer
# 
# tokenizer = RegexpTokenizer(r'\w+')
# print(tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!'))


with open("data/results5k_int.txt", 'w') as r5k:
    data = json.load(open("data/results5k.txt", 'r'))
    for row in data:
        if row['NSFW'] == 'True':
            print(row['title'])
        
