import numpy as np
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from __future__ import print_function

class Data: pass

class MyClassifier(object):
    def __init__(self):
        pass
    
    # read a tsv file 
    def read_tsv(self, fname):
        data = []
        labels = []
        f = open(fname, 'r')
        for line in f:
            (label,text) = line.strip().split("\t")
            labels.append(label)
            data.append(text)
        return data, labels
    
    # upload a train file (first column is label, second line is data)
    def upload_train_file(self, train_name):
        self.data = Data()
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.4, max_features=9000,smooth_idf=1)
    
        print("-- train data: ")
        self.data.train_data, self.data.train_labels = self.read_tsv(train_name)
        print(len(self.data.train_data))
        self.train_x = self.vectorizer.fit_transform(self.data.train_data)
        self.feature = self.vectorizer.get_feature_names()
        
        self.data.le = preprocessing.LabelEncoder()
        self.data.le.fit(self.data.train_labels)
        self.data.target_labels = self.data.le.classes_
        self.train_y = self.data.le.transform(self.data.train_labels)
        
        self.explainer = LimeTextExplainer(class_names = self.data.target_labels)

    # upload a test file (first column is label, second line is data)
    # before calling this, must call upload_train_file
    def upload_test_file(self, test_name):
        print("-- test data: ")
        self.data.test_data, self.data.test_labels = self.read_tsv(test_name)
        print(len(self.data.test_data))
        self.test_x = self.vectorizer.transform(self.data.test_data)
        self.test_y = self.data.le.transform(self.data.test_labels)
    
    def explain_sample(self, model, idx, sample, num_of_features):
        from sklearn.pipeline import make_pipeline
        c = make_pipeline(self.vectorizer, model)
        
        if not sample:
            mydata = self.data.test_data[idx]
        else:
            mydata = sample
        
        exp = self.explainer.explain_instance(mydata, c.predict_proba, num_features = num_of_features)
        if not sample:
            print('Sample id: %d' % idx)
            print('True class: %s' % self.data.target_labels[self.test_y[idx]])
        print('Predicted Class: %s' % self.data.target_labels[np.argmax(c.predict_proba([mydata]))])
        print('Probabilities = ',c.predict_proba([mydata]),'\n')
        print(exp.as_list(),"\n")
        # %matplotlib inline
        fig = exp.as_pyplot_figure()
        exp.show_in_notebook(text=True)
    

    def extract_features(self, data):
        return self.vectorizer.transform(data)

    def train_model(self):
        pass
    
    def predict_sample(self, sample):
        pass
    
    def predict_test(self):
        pass


class RandomForestClassifier(MyClassifier):
    def __init__(self):
        self.rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    
    # before calling this function, must call self.upload_train_file
    def train_model(self):
        self.rf.fit(self.train_x, self.train_y)

    # before calling this function, must call self.upload_test_file
    def predict_test(self):
        pred = self.rf.predict(self.test_x)
        print("The F1-score is: ")
        print(sklearn.metrics.f1_score(self.test_y, pred, average='binary'))

    def predict_sample(self, sample):
        pred = self.rf.predict([sample])
        print("The lable of this sample is :" + str(self.data.target_labels[pred[0]]))
    
    def explain_indexed_test_sample(self, idx, num_of_features = 6):
        self.explain_sample(self.rf, idx, None, num_of_features)
    
    def explain_self_input_sample(self, sample, num_of_features = 6):
        self.explain_sample(self.rf, -1, sample, num_of_features)
    
    def get_top_features(self, n = 10):
        vec = self.rf.feature_importances_
        tmp = []
        for i in range(len(vec)):
            tmp.append([vec[i], i])
        tmp.sort(reverse = True)
        print("The top-%d important Feature Name & Importance Value: \n" % n)
        for i in range(n):
            idx = tmp[i][1]
            importance_value = tmp[i][0]
            feature_name = self.feature[idx]
            print("%20s : %f \n " %(feature_name, importance_value))