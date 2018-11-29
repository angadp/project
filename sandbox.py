'''
Created on Nov 25, 2018
@author: ishank
'''
import json
# from nltk.tokenize import RegexpTokenizer
# 
# tokenizer = RegexpTokenizer(r'\w+')
# print(tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!'))

count = 0
nsfw_data, all_data = [], []
with open("data/results10k_int.txt", 'w') as r5k:
    
    data = json.load(open("data/results_nsfw.txt", 'r'))
    for row in data:
        if row['NSFW'] == 'True':
            nsfw_data.append({
                "title": str(row['title']),
                "NSFW": str(row['NSFW']),
                "text": str(row['text'])
            })
            
    data = json.load(open("data/results10k.txt", 'r'))
    for row in data:
        count += 1
        all_data.append({
            "title": str(row['title']),
            "NSFW": str(row['NSFW']),
            "text": str(row['text'])
        })
        
        if count % 48 == 0:
            if nsfw_data:
                all_data.append(nsfw_data.pop())
            
    r5k.write(json.dumps(all_data))
print(len(all_data))
            
