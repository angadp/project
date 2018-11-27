'''
Created on Nov 24, 2018
@author: ishank
'''

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
    from urllib.error import URLError, HTTPError
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen, URLError, HTTPError

import json
URL = "https://gateway.reddit.com/desktopapi/v1/frontpage"

class Reddit:
    def __init__(self, address):
        self._url = address
        self.post_ids = set()
        
    def fetch_data_dump(self):
        data = {}
        try:
            response = urlopen(self._url)
            data = json.load(response)        
        except HTTPError as e:
            print('The server couldn\'t fulfill the request. Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server. Reason: ', e.reason)       
        return data
    
    def get_data(self):
        
        data = self.fetch_data_dump()
        results = []
        
        if 'posts' in data:            
            for key, value in data['posts'].items():
                if key not in self.post_ids and 'media' in value:
                    self.post_ids.add(key)                     
                    if value['media'] and 'markdownContent' in value['media']:
                                         
                        result = {
                            'title': value['title'],
#                             'content': value['media']['markdownContent'],
                            'isNSFW': value['isNSFW']
                        }
                        results.append(result)
                        print(result)
        
    
        return results


reddit = Reddit(URL)

# while True:
reddit.get_data()