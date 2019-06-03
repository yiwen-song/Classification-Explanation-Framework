import numpy as np
import csv
from sklearn.model_selection import train_test_split

csvFile = open("gender-classifier-DFE-791531.csv", "r",encoding='utf-8',errors='ignore')
reader = csv.reader(csvFile)

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
            description.append(item[10].replace('\n',' '))
            text.append(item[19].replace('\n',' '))
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

