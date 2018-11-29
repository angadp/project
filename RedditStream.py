'''
Created on Nov 24, 2018
@author: ishank
'''

from __future__ import print_function
import praw, json
NUM_REDDITS = 200
DEBUG = True
from random import randint

class RedditStream:
    def __init__(self, subreddit='all'):
        self.reddit = None
        self.subreddit = subreddit
        self.count = 0
        self.authenticate()
        
    def authenticate(self):
        config = json.load(open('app.json'))
        
        try:
            self.reddit = praw.Reddit(client_id=config['client_id'],
                                 client_secret=config['client_secret'],
                                 password=config['password'],
                                 user_agent=config['user_agent'],
                                 username=config['username'])

            print('Logged in: '+ str(self.reddit.user.me()))
        except praw.exceptions.PRAWException as e:
            print(e)
        except Exception as e:
            print(e)
        
    def stream_data(self):
        results = []
        for submission in self.reddit.subreddit(self.subreddit).stream.submissions():
            if not submission.stickied and submission.selftext:                
                try:
                    if submission.over_18:                
                        results.append({
                            "title": self.cleanse(submission.title.decode('utf-8')),
                            "text": self.cleanse(submission.selftext.decode('utf-8')),
                            "NSFW": str(submission.over_18)
                        })                    
                        if DEBUG and self.count % 10 == 0: # randint(1,50) == 1:
                            title = ("\t(NSFW) " if submission.over_18 else "\t(SFW)  ") + submission.title
                            print(self.count, " " , title[:40] + ("..." if len(title) > 40 else ""))                   
                        self.count += 1
                except:
                    pass
                
            if self.count >= NUM_REDDITS: break
        return results
    
    def cleanse(self, text):
        text = text.strip().replace('_', ' ').replace("\\\\", "\\").replace('\n', ' ').replace('\\n', ' ')
        return str(text.lower().decode('unicode-escape'))
        
#         return str(' '.join(filter(lambda x: x.isalnum(), text.split())))
    
    @staticmethod
    def jsonify(item):            
        return str(item) + ","

def main():
    redstr = RedditStream()
    
    with open("results.txt", "w") as res:
#         res.write("[")
        results = []
        for item in redstr.stream_data():
            results.append(item)
        res.write(json.dumps(results))
#             res.write(RedditStream.jsonify(item))
#         res.write("]")
    if DEBUG:
        print("\nCollected: ", redstr.count)

if __name__ == "__main__":
    main()     