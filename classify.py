from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import spacy
import numpy as np
import time

nlp = spacy.load('en')
with open('affirmations.txt') as f:
    affirmations = f.read().split('\n')

class PredictQuestionType(object):
    def __init__(self, documents=None, document_path=None):
        self.special_features = []
        if documents:
            raw_texts = documents
        elif document_path:
            try:
                with open(document_path) as f:
                    raw_texts = [line.strip() for line in f.readlines()]
            except IOError:
                raw_texts = None
        else:
            raw_texts = None

        if raw_texts is None:
            raise IOError("No Documents found.")
        self.sents = pd.DataFrame(raw_texts, columns=['sentence'])

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

    def featurize(self):
        self.sents.sentence.apply(self.get_sent_features)

        print 'Loading vectorizing models'
        dict_vect = joblib.load('models/dict_vect.pkl')
        tfidf_vect = joblib.load('models/tfidf_vect.pkl')

        print 'Transforming documents to feature vectors'
        spec_features = dict_vect.transform(self.special_features)
        tfidf_features = tfidf_vect.transform(self.sents.sentence).toarray()

        self.X_test = np.concatenate((spec_features, tfidf_features), axis=1)


    def label_decoder(self, labels):
        lencoder = joblib.load('models/label_encoder.pkl')
        return lencoder.inverse_transform(labels)

    def predict(self):
        print 'Creating feature vector'
        self.featurize()

        print 'Loading model from directory'
        model = joblib.load('models/GradientBoostingClassifier.pkl')

        print 'Predicting Question Type'
        self.sents['predicted_cat'] = self.label_decoder(model.predict(self.X_test))

        print '#'*30
        print "Predictions"
        for index, row in self.sents.iterrows():
            print 'Question :', row.sentence.strip()
            print 'Predicted Category :', row.predicted_cat

        fname = 'predicted_test_'+str(int(time.time()))+'.csv'
        print 'Saving the output to :', fname
        self.sents.to_csv(fname, index=False)


while True:
    try:
        print 'Do you have a document file? (y/n) (Press Ctrl+C to exit)'
        prompt = str(raw_input())
        if prompt == 'y':
            document_path = str(raw_input("Please enter the full document path :\n"))
            predictor = PredictQuestionType(document_path=document_path)
            predictor.predict()
        else:
            document = str(raw_input("Please enter your sentence :\n"))
            predictor = PredictQuestionType(document_path=[document])
            predictor.predict()
    except KeyboardInterrupt:
        print 'Exiting.'
        break
