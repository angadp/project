'''
Created on Nov 27, 2018
@author: ishank
'''
import numpy as np

class Classifier():
    """Parent class for classifiers"""
    
    def train(self, texts, labels):
        """Learn the parameters of the model from the given labeled data."""
        pass
    
    def predict(self, texts): 
        """Make predictions using the learned model"""
        pass


class LinearSVM(Classifier):
    def __init__(self, alpha=11e-5):
        self.classifier = None
        self.alpha = alpha

    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        # best alpha=5e-5, n_iter=15, penalty='l2', random_state=42
        clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, min_df=5, analyzer="word", ngram_range=(1, 3))),
                         ('clf', SGDClassifier(alpha=self.alpha, n_iter=15, penalty='l2', random_state=42))])
        self.classifier = clf.fit(texts, labels)

    def predict(self, words):
        assert self.classifier != None
        return self.classifier.predict(words)

class word2Vec(Classifier):
    def __init__(self):
        self.clf = None

    def buildWordVector(self, w2vmodel, text, size):
        import nltk
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        stops = set(nltk.corpus.stopwords.words('english'))
        for word in text.split(' '):
            if word not in stops:
                try:
                    sorted_vec = np.sort(w2vmodel[word])
                    vec += sorted_vec.reshape((1, size))
                    count += 1.
                except KeyError:
                    continue
            if count != 0:
                vec /= count
        return vec

    def w2vectorize(self, collection, model, n_dim):
        from sklearn.preprocessing import scale
        vecs = np.concatenate([self.buildWordVector(model, z, n_dim) for z in collection])
        vecs = scale(vecs)

        return vecs

    def train(self, texts, labels):
        import nltk
        from gensim.models import Word2Vec
        from sklearn.linear_model import SGDClassifier
        tok = []
        for text in texts:
            tokens = nltk.wordpunct_tokenize(text)
            stops = set(nltk.corpus.stopwords.words('english'))
            tok.append([token for token in tokens if token not in stops])
        self.model = Word2Vec(tok, window=5, min_count=3, workers=4)
        model = Word2Vec(tok, window=5, min_count=3, workers=4)
        train_vecs = self.w2vectorize(texts, model, 100)
        classifier = SGDClassifier(loss='log', penalty='l2')
        classifier.fit(train_vecs, labels)

        self.classifier = classifier


    def predict(self, words):
        assert self.classifier != None
        test_vecs = self.w2vectorize(words, self.model, 100)
        predictions = self.classifier.predict(test_vecs)
        return predictions

class LogisticRegressionClassifier(Classifier):
    
    def __init__(self):
        self.classifier = None
        
    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline 
        from sklearn.linear_model import LogisticRegressionCV
        
        word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=5)
        clf = LogisticRegressionCV(tol=1e-9, penalty='l2', random_state=0, solver='lbfgs', max_iter=2000)
        
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
        from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        rdf_clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=5)),
                      ('clf', RandomForestClassifier(random_state=42, n_estimators=15))])        
        
        adb_clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=5)),
                      ('clf', AdaBoostClassifier(random_state=42, learning_rate=0.6, n_estimators=100))])
                
        log_clf = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=5)),
                      ('clf', LogisticRegression(tol=1e-9, penalty='l2', C=5))])
        
        vot_clf = VotingClassifier(estimators=[('svm', adb_clf), ('log', log_clf), ('rdf', rdf_clf)],
                                        voting='soft', weights=[1,7,1])
        
        self.classifier = vot_clf.fit(texts, labels)
        return self.classifier
        
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
        
class AdaboostClassifier(Classifier):
    
    def __init__(self):
        self.classifier = None
        
    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline 
        from sklearn.ensemble import AdaBoostClassifier

        classifier = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=5)),
                      ('clf', AdaBoostClassifier(random_state=42, learning_rate=0.6, n_estimators=100))])
                
        self.classifier = classifier.fit(texts, labels)
        return self.classifier
        
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
    
class RandomForestClassifier(Classifier):
    
    def __init__(self):
        self.classifier = None
        
    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline 
        from sklearn.ensemble import RandomForestClassifier

        classifier = Pipeline([('word_vec', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=5)),
                      ('clf', RandomForestClassifier(random_state=42, n_estimators=15))])
                
        self.classifier = classifier.fit(texts, labels)
        return self.classifier
        
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
        
class NaiveClassifier(Classifier):
    def __init__(self):
        self.classifier = None
        self.nsfw_words = set([])
    
    def train(self, texts, labels):
        with open('badwords.txt', 'r') as f:
            x = f.readlines()
                        
        self.nsfw_words = set(list(map(str.strip, x)))
        
    def predict(self, texts):                
        predicted = []
        for i in texts:
            currentWords = i.lower().split(" ")
            if len(set(currentWords).intersection(self.nsfw_words)) > 0:
                predicted.append(1)
            else:
                predicted.append(0)
        return predicted        
