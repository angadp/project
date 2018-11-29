'''
Created on Nov 27, 2018
@author: ishank
'''

class Classifier():
    """Parent class for classifiers"""
    
    def train(self, texts, labels):
        """Learn the parameters of the model from the given labeled data."""
        pass
    
    def predict(self, texts): 
        """Make predictions using the learned model"""
        pass


class LinearSVM(Classifier):
    def __init__(self):
        self.classifier = None

    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, min_df=3, analyzer="word", ngram_range=(1, 3))),
                         ('clf', SGDClassifier(alpha=1e-3, n_iter=10, penalty='l2', random_state=42))])
        self.classifier = clf.fit(texts, labels)

    def predict(self, words):
        assert self.classifier != None
        return self.classifier.predict(words)

class LogisticRegressionClassifier(Classifier):
    
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
    
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
        
class EnsembleVotingClassifier(Classifier):
    
    def __init__(self):
        self.classifier = None
        
    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline 
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import SGDClassifier, LogisticRegression
        
        svm_clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
        
        log_clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)),
                      ('clf', LogisticRegression(tol=1e-8, penalty='l2', C=4))])
        
        vot_clf = VotingClassifier(estimators=[('svm', svm_clf), ('log', log_clf)],
                                        voting='soft', weights=[1,6])
        
        self.classifier = vot_clf.fit(texts, labels)
        return self.classifier
        
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
        
        
        
        
        
        
        
        
            
    

