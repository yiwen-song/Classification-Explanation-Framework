import numpy as np
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
# from __future__ import print_function

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
        stopWords = list((stopwords.words('english')))
        self.vectorizer = TfidfVectorizer(stop_words = stopWords, ngram_range=(1,2), min_df=2, \
                                          max_df=0.4, max_features=9000,smooth_idf=1)
    
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

    def extract_features(self, data):
        return self.vectorizer.transform(data)

    def train_model(self):
        pass
    
    def predict_test(self):
        pass
    
    def explain_self_input_sample(self, sample):
        pass
    
    def explain_indexed_test_sample(self, idx):
        pass
    
    def get_top_features(self, n = 10):
        pass

    def get_roc(self):
        pass


class RandomForestClassifier(MyClassifier):
    def __init__(self):
        self.rf = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
    
    # before calling this function, must call self.upload_train_file
    def train_model(self):
        print ("train RandomForestClassifier")
        self.rf.fit(self.train_x, self.train_y)

    # before calling this function, must call self.upload_test_file
    def predict_test(self):
        y_pred = self.rf.predict(self.test_x)
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(\
            self.test_y, y_pred, average = None, labels = [0,1])
        print("The precision is: ", precision)
        print("The recall is: ", recall)
        print("The f1-score is: ", fscore)
        data = []
        data.append(precision)
        data.append(recall)
        data.append(fscore)
        data = np.array(data).T  
        return self.data.target_labels, data

    def get_roc(self):
        y_score = self.rf.predict_proba(self.test_x)
        fpr, tpr, threshold = roc_curve(self.test_y, y_score[:,1]) 
        roc_auc = auc(fpr,tpr)
        fpr = np.around(fpr, decimals = 4)
        tpr = np.around(tpr, decimals = 4)
        print("roc_auc:",roc_auc)
        label = 'ROC Curve (Area = %0.3f)' % roc_auc
        return list(fpr), list(tpr), label
    
    def explain_indexed_test_sample(self, idx, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.rf)
        mydata = self.data.test_data[idx]
        exp = self.explainer.explain_instance(mydata, c.predict_proba, num_features = num_of_features)
        print('Sample id: %d' % idx)
        print('True class: %s' % self.data.target_labels[self.test_y[idx]])
        print('Predicted Class: %s' % self.data.target_labels[np.argmax(c.predict_proba([mydata]))])
        print('Probabilities = ',c.predict_proba([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)
    
    def explain_self_input_sample(self, sample, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.rf)
        mydata = sample
        exp = self.explainer.explain_instance(mydata, c.predict_proba, num_features = num_of_features)
        print('Predicted Class: %s' % self.data.target_labels[np.argmax(c.predict_proba([mydata]))])
        print('Probabilities = ',c.predict_proba([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)
    
    def get_top_features(self, n = 10):
        vec = self.rf.feature_importances_
        print(vec)
        tmp = []
        for i in range(len(vec)):
            tmp.append([abs(vec[i]),i,vec[i]])
        tmp.sort(reverse = True)
        print("The top-%d important Feature Name & Importance Value: \n" % n)
        datas = []
        labels = []
        for i in range(n):
            idx = tmp[i][1]
            importance_value = tmp[i][2]
            feature_name = self.feature[idx]
            datas.append(importance_value)
            labels.append(feature_name)
            # print("%20s : %f \n " %(feature_name, importance_value))
        return datas, labels


class LogisticRegressionClassifier(MyClassifier):
    def __init__(self):
        self.lr = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, C=1.5)
    
    # before calling this function, must call self.upload_train_file
    def train_model(self):
        print ("train LogisticRegressionClassifier")
        self.lr.fit(self.train_x, self.train_y)

    def predict_test(self):
        y_pred = self.lr.predict(self.test_x)
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(\
            self.test_y, y_pred, average = None, labels = [0,1])
        print("The precision is: ", precision)
        print("The recall is: ", recall)
        print("The f1-score is: ", fscore)
        data = []
        data.append(precision)
        data.append(recall)
        data.append(fscore)
        data = np.array(data).T  
        return self.data.target_labels, data

    def get_roc(self):
        y_score = self.lr.predict_proba(self.test_x)
        fpr, tpr, threshold = roc_curve(self.test_y, y_score[:,1]) 
        roc_auc = auc(fpr,tpr)
        fpr = np.around(fpr, decimals = 4)
        tpr = np.around(tpr, decimals = 4)
        print("roc_auc:",roc_auc)
        label = 'ROC Curve (Area = %0.3f)' % roc_auc
        return list(fpr), list(tpr), label
    
    def explain_indexed_test_sample(self, idx, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.lr)
        mydata = self.data.test_data[idx]
        exp = self.explainer.explain_instance(mydata, c.predict_proba, num_features = num_of_features)
        print('Sample id: %d' % idx)
        print('True class: %s' % self.data.target_labels[self.test_y[idx]])
        print('Predicted Class: %s' % self.data.target_labels[np.argmax(c.predict_proba([mydata]))])
        print('Probabilities = ',c.predict_proba([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)
    
    def explain_self_input_sample(self, sample, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.lr)
        mydata = sample
        exp = self.explainer.explain_instance(mydata, c.predict_proba, num_features = num_of_features)
        print('Predicted Class: %s' % self.data.target_labels[np.argmax(c.predict_proba([mydata]))])
        print('Probabilities = ',c.predict_proba([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)

    def get_top_features(self, n = 10):
        vec = abs(self.lr.coef_[0])
        tmp = []
        for i in range(len(vec)):
            tmp.append([vec[i], self.lr.coef_[0][i], i])
        tmp.sort(reverse = True)
        print("The top-%d important Feature Name & Importance Value: \n" % n)
        datas = []
        labels = []
        for i in range(n):
            idx = tmp[i][2]
            importance_value = tmp[i][1]
            feature_name = self.feature[idx]
            datas.append(importance_value)
            labels.append(feature_name)
            print("%20s : %f \n " %(feature_name, importance_value))
        return datas,labels

class SVMClassifier(MyClassifier):
    def __init__(self):
        self.svm = LinearSVC(random_state=0, tol=1e-5)

    # before calling this function, must call self.upload_train_file
    def train_model(self):
        self.svm.fit(self.train_x, self.train_y)

    def predict_test(self):
        y_pred = self.svm.predict(self.test_x)
        precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(\
            self.test_y, y_pred, average = None, labels = [0,1])
        print("The precision is: ", precision)
        print("The recall is: ", recall)
        print("The f1-score is: ", fscore)
        data = []
        data.append(precision)
        data.append(recall)
        data.append(fscore)
        data = np.array(data).T  
        return self.data.target_labels, data
    
    def get_roc(self):
        y_score = self.svm.decision_function(self.test_x)
        fpr, tpr, threshold = roc_curve(self.test_y, y_score) 
        roc_auc = auc(fpr,tpr)
        fpr = np.around(fpr, decimals = 4)
        tpr = np.around(tpr, decimals = 4)
        print("roc_auc:",roc_auc)
        label = 'ROC Curve (Area = %0.3f)' % roc_auc
        return list(fpr), list(tpr), label
    
    def predict_probs(self,datas):
        res = []
        for data in datas:
            prob = self.svm.decision_function([data])[0]
            if(self.svm.decision_function([data])[0] == 0):
                res.append([prob,max(0,1-prob)])
            else:
                res.append([max(1-prob,0),prob])
        return np.array(res)

    def explain_indexed_test_sample(self, idx, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.svm)
        mydata = self.data.test_data[idx]
        exp = self.explainer.explain_instance(mydata, self.predict_probs, num_features = num_of_features)
        print('Sample id: %d' % idx)
        print('True class: %s' % self.data.target_labels[self.test_y[idx]])
        print('Predicted Class: %s' % self.data.target_labels[c.predict([mydata])[0]])
        print('Probabilities = ',c.decision_function([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)
    
    def explain_self_input_sample(self, sample, num_of_features = 6, savefile='./tmp/local_res.html'):
        c = make_pipeline(self.vectorizer, self.svm)
        mydata = sample
        exp = self.explainer.explain_instance(mydata, self.predict_probs, num_features = num_of_features)
        print('Predicted Class: %s' % self.data.target_labels[c.predict([mydata])[0]])
        print('Probabilities = ',c.decision_function([mydata]),'\n')
        print(exp.as_list(),"\n")
        exp.save_to_file(savefile)
    
    def get_top_features(self, n = 10):
        vec = abs(self.svm.coef_[0])
        tmp = []
        for i in range(len(vec)):
            tmp.append([vec[i], self.svm.coef_[0][i], i])
        tmp.sort(reverse = True)
        print("The top-%d important Feature Name & Importance Value: \n" % n)
        datas = []
        labels = []
        for i in range(n):
            idx = tmp[i][2]
            importance_value = tmp[i][1]
            feature_name = self.feature[idx]
            datas.append(importance_value)
            labels.append(feature_name)
            print("%20s : %f \n " %(feature_name, importance_value))
        return datas,labels


if __name__ == "__main__":
    rf_classifier = RandomForestClassifier()
    rf_classifier.upload_train_file("train.tsv")
    rf_classifier.upload_test_file("dev.tsv")
    rf_classifier.train_model()
    rf_classifier.predict_test()
    rf_classifier.explain_indexed_test_sample(100)
