from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import spacy
import numpy as np
import get_affirmations

nlp = spacy.load('en')
with open('affirmations.txt') as f:
    affirmations = f.read().split('\n')

class TrainModel(object):
    def __init__(self, filepath):
        df = pd.read_table(filepath, delimiter=" ,,, ", names=['sentence', 'category'])
        self.sents = df.sentence
        self.cat = df.category
        self.special_features = []

    def get_sent_features(self, sent):
        text = nlp(unicode(sent, encoding='utf-8'))
        s_feature = {
            'tag':"",
            'is_wh':False,
            'is_affirmation':False
        }
        for token in text:
            if token.text.lower().startswith('wh'):
                s_feature['is_wh'] = True
            if token.text.lower() in affirmations:
                s_feature['is_affirmation'] = True
            s_feature['tag'] = token.tag_
            break
        self.special_features.append(s_feature)

    def label_encoder(self):
        lencoder = LabelEncoder().fit(self.cat)
        print 'Dumping Label Encoder to models directory'
        joblib.dump(lencoder, 'models/label_encoder.pkl')
        return lencoder

    def feature_extraction(self):
        #Get Special Features
        self.sents.apply(self.get_sent_features)
        special_vect = DictVectorizer(sparse=False)
        X_spec = special_vect.fit_transform(self.special_features)
        print 'Dumping Dict Vectorizer to models directory'
        joblib.dump(special_vect, 'models/dict_vect.pkl')

        #Get TfIdf Vector
        tfidf_vect = TfidfVectorizer(ngram_range=[1, 2], encoding='utf-8')
        X_tfidf = tfidf_vect.fit_transform(self.sents).toarray()
        print 'Dumping TfIdf vectorizer to models directory'
        joblib.dump(tfidf_vect, 'models/tfidf_vect.pkl')

        #Merge 2 feature vectors
        self.X = np.concatenate((X_spec, X_tfidf), axis=1)
        print self.X.shape

    def train(self):
        #Extract Features
        print 'Extracting Features'
        self.feature_extraction()
        #Pre Processing
        print 'Encoding Labels'
        lencoder = self.label_encoder()

        print 'Training Model'
        model = GradientBoostingClassifier(random_state=42)#Life Universe and Everything
        y = lencoder.transform(self.cat)

        model.fit(self.X, y)
        print 'Training Complete, dumping model to models directory'
        joblib.dump(model, 'models/GradientBoostingClassifier.pkl')


#model = TrainModel('data/LabelledData (1).txt')
#model.train()

