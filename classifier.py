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
    def __init__(self):
        self.clf = None

    def train(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import Pipeline

        text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, min_df=3, analyzer="word", ngram_range=(1, 3))),
                         ('clf', SGDClassifier(alpha=1e-3, n_iter=5, penalty='l2', random_state=42))])
        self.clf = text_clf.fit(texts, labels)

    def predict(self, words):
        assert self.classifier != None
        return self.clf.predict(words)

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
        classifier = SGDClassifier(loss='log', penalty='l1')
        classifier.fit(train_vecs, labels)

        self.classifier =classifier


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
        from sklearn.linear_model import LogisticRegression
        
        word_vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
        clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
        
        classifier = Pipeline([('word_vec', word_vec), ('clf', clf)])
        self.classifier = classifier.fit(texts, labels)
        
        return self.classifier
    
    def predict(self, texts):        
        assert self.classifier != None
        return self.classifier.predict(texts)
        
class NaiveClassifier(Classifier):
    def __init__(self):
        self.classifier = None
        
    def predict(self,texts,labels):
        with open('badwords.txt', 'r') as f:
            x = f.readlines()
            #to remove \n from list elements    
        b_words=list(map(str.strip,x))
        predicted = []
        for i in texts:
            currentWords = i.lower().split(" ")
            if len(set(currentWords).intersection(set(b_words))) > 0:
                predicted.append(1)
            else:
                predicted.append(0)
        return(predicted)        
