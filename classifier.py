'''
Created on Nov 27, 2018
@author: ishank
'''

class LogisticRegressionClassifier():
    
    def __init__(self):
        self.classifier = None
        
    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline 
        from sklearn.linear_model import LogisticRegression
        
        word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
        clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
        
        classifier = Pipeline([('word_vec', word_vec), ('clf', clf)])
        self.classifier = classifier.fit(texts, labels)
        
        return self.classifier
    
    def predict(self, text):        
        assert self.classifier != None
        return self.classifier.predict(text)
        
            
    

