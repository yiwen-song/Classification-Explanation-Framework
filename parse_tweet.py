import numpy as np
import csv
import re
from sklearn.model_selection import train_test_split

csvFile = open("gender-classifier-DFE-791531.csv", "r",encoding='utf-8',errors='ignore')
reader = csv.reader(csvFile)

def remove_url(s):
    url_reg = r'https://[a-zA-Z0-9.?/&=:]*'
    s = re.sub(url_reg, '', s)
    return s

def cleaning(s):
    s = str(s)
    s = s.lower()
    s = remove_url(s)
    s = s.replace("'s",' is')
    s = s.replace("'re",' are')
    s = s.replace("'ve",' have')
    s = s.replace("'m",' am')
    s = re.sub('\s\W',' ',s)         #whitespace characters
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", '', s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_]', '', s)
    s = s.replace("ù","")
    s = s.replace("ù", "")
    s = s.replace("û", "")
    s = s.replace("âù", "")
    s = s.replace("ü", "")
    s = s.replace("å", "")
    s = s.replace("â", "")
    s = s.replace("ä", "")
    s = s.replace("ή", "")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    return s


# read messages:
gender = []
text = []
description = []
confidence = []
for item in reader:
    if reader.line_num == 1:
        continue
    if ((item[5] == 'female') or (item[5] == 'male')):
        confidence = float(item[6])
        if confidence >= 0.9: 
            gender.append(item[5])
            # description.append(item[10].replace('\n',''))
            # text.append(item[19].replace('\n',''))
            description.append(cleaning(item[10]))
            text.append(cleaning(item[19]))
csvFile.close()

n = len(gender)
print("we have %d data in total.\n"%n)

# data = []
data = text
label = gender
# for i in range(n):
#     data.append(description[i] + ' ' + text[i])

# print(data[0])
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)
n_train = len(X_train)
n_test = len(X_test)
f1 = open("train-tweeter.tsv","w")
for i in range(n_train):
    f1.write(str(y_train[i]) + '\t' + str(X_train[i]) + '\n')
f1.close()

f2 = open("test-tweeter.tsv","w")
for i in range(n_test):
    f2.write(str(y_test[i]) + '\t' + str(X_test[i]) + '\n')
f2.close()

# x = str(y_train[0]) + '\t' + str(X_train[0]) + '\n'
# a,b = x.strip().split('\t')
# print(a,'\n',b)

