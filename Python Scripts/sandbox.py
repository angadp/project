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
with open("final.json", 'w') as r5k:
    
#     data = json.load(open("data/results_nsfw.txt", 'r'))
#     for row in data:
#         if row['NSFW'] == 'True':
#             nsfw_data.append({
#                 "title": str(row['title']),
#                 "NSFW": str(row['NSFW']),
#                 "text": str(row['text'])
#             })
    final = {
        "description": ["Label headlines of Reddit posts as NSFW or not"],
        "authors": {
                "author1": "Ishank Mishra",
                "author2": "Angadpreet Nagpal",
                "author3": "Somesh Sakriya"
            },
        "emails": {
                "email1": "imishra@usc.edu",
                "email2": "asnagpal@usc.edu",
                "email3": "sakriya@usc.edu"
            }
    }
    
    data = json.load(open("data/results10k.txt", 'r'))
    for row in data:
#         count += 1
        lab = "NSFW" if row['NSFW'] == "True" else "SFW"
        all_data.append({
            "label": str(lab),
            "data": str(row['text'])
        })
    
    final['corpus'] = all_data
    
#         if count % 48 == 0:
#             if nsfw_data:
#                 all_data.append(nsfw_data.pop())
            
    r5k.write(json.dumps(final))
    
print(len(all_data))


            
