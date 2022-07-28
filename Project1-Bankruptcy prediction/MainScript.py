#!/usr/bin/env python
# coding: utf-8

# ## Bankruptcy prediction project

# Asimina Tzana 
# AIVC21015
# aivc21015@uniwa.gr
# Part 2

# In[1]:


#import libraries
import os

import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import xlrd
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn import under_sampling

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense


# In[2]:


# read the data
try:
 # Confirm file exists.
 df = pd.read_excel("InputData\\Dataset.xlsx")
 print("Column headings:")
 print(df.columns)
except FileNotFoundError:
 print(FileNotFoundError)


# In[3]:


#Plot original data
fig = px.pie(df, values='ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)', names='ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)')
fig.show()


# In[4]:


inputData= df[df.columns[0:10]].values


# In[5]:


outputData = df['ΕΝΔΕΙΞΗ ΑΣΥΝΕΠΕΙΑΣ (=2) (ν+1)']
outputData , levels=pd.factorize(outputData)


# In[6]:


print(' .. we have', inputData.shape[0], 'available paradigms.')
print(' .. each paradigm has', inputData.shape[1], 'features')

print(' ... the distribution for the available class lebels is:')
for classIdx in range(0, len(np.unique(outputData))):
    tmpCount = sum(outputData == classIdx)
    tmpPercentage = tmpCount/len(outputData)
    print(' .. class', str(classIdx), 'has', str(tmpCount), 'instances', '(', '{:.2f}'.format(tmpPercentage), '%)')


# In[7]:


#Split data into Training and Testing Sets 
X_train, X_test, y_train, y_test = train_test_split(inputData, outputData, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


#Perform undersample techique
failed=0
non_failed=0
for i in range(len(y_train)):
    if y_train[i]==1:
        failed=failed+1
print("Companies That Went Bankrupt in training set",failed)
for i in range(len(y_train)):
    if y_train[i]==0:
        non_failed=non_failed+1
print("Healthy companies in training",non_failed)

synolo2=0
synolo1=0
for i in range(len(y_test)):
    if y_test[i]==1:
        synolo2=synolo2+1
print("Healthy companies in test set",synolo2)
for i in range(len(y_test)):
    if y_test[i]==0:
        synolo1=synolo1+1
print("Companies That Went Bankrupt in test set",synolo1)


#analogy 3:1 for training set
rus = under_sampling.RandomUnderSampler(
    sampling_strategy={
        0: failed*3,
        1: failed,

    },
    random_state=42
)

failed=0
non_failed=0

#analogy 3:1 for test set
rus2 = under_sampling.RandomUnderSampler(
    sampling_strategy={
        0: synolo2*3,
        1: synolo2,

    },
    random_state=42
)


# In[9]:


X_train_new,y_train_new=rus.fit_resample(X_train,y_train)
X_Test_new,y_test_new=rus2.fit_resample(X_test,y_test)


# In[10]:


failed=0
non_failed=0
for i in range(len(y_train_new)):
    if y_train_new[i]==1:
        failed=failed+1
print("Healthy companies in training set",failed)
for i in range(len(y_train_new)):
    if y_train_new[i]==0:
        non_failed=non_failed+1
print("Healthy companies in training set",non_failed)

synolo2=0
synolo1=0
for i in range(len(y_test_new)):
    if y_test_new[i]==1:
        synolo2=synolo2+1
print("Healthy companies in test set",synolo2)
for i in range(len(y_test_new)):
    if y_test_new[i]==0:
        synolo1=synolo1+1
print("Healthy companies in test set",synolo1)

non_healthy_training = failed
non_healthy_test = synolo2


# In[11]:


y_train_new_df = pd.DataFrame(y_train_new, columns = ['Label'])
y_train_labels = y_train_new_df.value_counts()
#Plot new data
fig = px.pie(y_train_new_df, names='Label', title='Data after undersampling with 3:1')
fig.show()


# ### Linear Discriminant Analysis

# In[12]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_new, y_train_new) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
lda_y_pred_train = lda.predict(X_train_new)
lda_y_pred_test = lda.predict(X_Test_new)


# In[13]:


#calculate the scores
# now check for both train and test data, how well the model learned the patterns
acc_train_lda = accuracy_score(y_train_new, lda_y_pred_train)
acc_test_lda = accuracy_score(y_test_new, lda_y_pred_test)
pre_train_lda = precision_score(y_train_new, lda_y_pred_train,zero_division = 0, average='binary')
pre_test_lda = precision_score(y_test_new, lda_y_pred_test, average='binary')
rec_train_lda = recall_score(y_train_new, lda_y_pred_train, average='binary')
rec_test_lda = recall_score(y_test_new, lda_y_pred_test, average='binary')
f1_train_lda = f1_score(y_train_new, lda_y_pred_train, average='binary')
f1_test_lda = f1_score(y_test_new, lda_y_pred_test, average='binary')


# In[14]:


# print the scores
print('Accuracy scores of Linear Discriminant Analysis classifier are:','train: {:.2f}'.format(acc_train_lda), 'and test:{:.2f}.'.format(acc_test_lda))
print('Precision scores of Linear Discriminant Analysis classifier are:','train: {:.2f}'.format(pre_train_lda), 'and test:{:.2f}.'.format(pre_test_lda))
print('Recall scores of Linear Discriminant Analysis classifier are:','train: {:.2f}'.format(rec_train_lda), 'and test:{:.2f}.'.format(rec_test_lda))
print('F1 scores of Linear Discriminant Analysis classifier are:','train: {:.2f}'.format(f1_train_lda), 'and test: {:.2f}.'.format(f1_test_lda))


# In[15]:


#Classification report for test set
print('                 LDA Test set classification report')
print(classification_report(y_test_new, lda_y_pred_test))


# In[16]:


#Classification report for train set
print('                 LDA Train set classification report')
print(classification_report(y_train_new, lda_y_pred_train))


# In[17]:


plt.rcParams["figure.figsize"] = (12,6)


plt.scatter(X_Test_new[lda_y_pred_test==0, 0] , X_Test_new[lda_y_pred_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[lda_y_pred_test==1, 0] , X_Test_new[lda_y_pred_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('Linear Discriminant Analysis')
plt.show()


# In[18]:


# Creates a confusion matrix
cm_lda_train = confusion_matrix(y_train_new, lda_y_pred_train)
#plots the confusion matrix

plt.figure(figsize=(6,5))
sns.heatmap(cm_lda_train, annot=True, fmt=".1f")
plt.title('Linear Discriminant Analysis Train\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, lda_y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.show()


# In[19]:


# Creates a confusion matrix
cm_lda_test = confusion_matrix(y_test_new, lda_y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_lda_test, annot=True, fmt=".1f")
plt.title('Linear Discriminant Analysis Test\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, lda_y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[20]:


#Print TP FP FN TN
#training 
TP_lda_train = cm_lda_train[1][1]
FP_lda_train = cm_lda_train[1][0]
FN_lda_train = cm_lda_train[0][1]
TN_lda_train = cm_lda_train[0][0]
print('TP is : {:.2f}.'.format(TP_lda_train))
print('FP is : {:.2f}.'.format(FP_lda_train))
print('FN is : {:.2f}.'.format(FN_lda_train))
print('TN is : {:.2f}.'.format(TN_lda_train))

#test
TP_lda_test = cm_lda_test[1][1]
FP_lda_test = cm_lda_test[1][0]
FN_lda_test = cm_lda_test[0][1]
TN_lda_test = cm_lda_test[0][0]
print('TP is : {:.2f}.'.format(TP_lda_test))
print('FP is : {:.2f}.'.format(FP_lda_test))
print('FN is : {:.2f}.'.format(FN_lda_test))
print('TN is : {:.2f}.'.format(TN_lda_test))


# In[21]:


#Now the normalize the diagonal entries
cm_lda_full = cm_lda_test.astype('float') / cm_lda_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_lda_full.diagonal()


# In[22]:


#Now the normalize the diagonal entries
cm_lda_full = cm_lda_train.astype('float') / cm_lda_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_lda_full.diagonal()


# ### Logistic Regression

# In[23]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_new, y_train_new) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
log_y_pred_train = logreg.predict(X_train_new)
log_y_pred_test = logreg.predict(X_Test_new)


# In[24]:


#calculate the scores
# now check for both train and test data, how well the model learned the patterns
acc_train_log = accuracy_score(y_train_new, log_y_pred_train)
acc_test_log = accuracy_score(y_test_new, log_y_pred_test)
pre_train_log = precision_score(y_train_new, log_y_pred_train,zero_division = 0, average='macro')
pre_test_log = precision_score(y_test_new, log_y_pred_test,zero_division = 0, average='macro')
rec_train_log = recall_score(y_train_new, log_y_pred_train, average='macro')
rec_test_log = recall_score(y_test_new, log_y_pred_test, average='macro')
f1_train_log = f1_score(y_train_new, log_y_pred_train, average='macro')
f1_test_log = f1_score(y_test_new, log_y_pred_test, average='macro')


# In[25]:


# print the scores
print('Accuracy scores of Logistic Regression classifier are:','train: {:.2f}'.format(acc_train_log), 'and test:{:.2f}.'.format(acc_test_log))
print('Precision scores of Logistic Regression classifier are:','train: {:.2f}'.format(pre_train_log), 'and test:{:.2f}.'.format(pre_test_log))
print('Recall scores of Logistic Regression classifier are:','train: {:.2f}'.format(rec_train_log), 'and test:{:.2f}.'.format(rec_test_log))
print('F1 scores of Logistic Regression classifier are:','train: {:.2f}'.format(f1_train_log), 'and test: {:.2f}.'.format(f1_test_log))


# In[26]:


print('                LR Train set classification report')
print(classification_report(y_train_new, log_y_pred_train))


# In[27]:


print('                LR Test set classification report')
print(classification_report(y_test_new, log_y_pred_test))


# In[28]:


# Creates a confusion matrix
cm_log_train = confusion_matrix(y_train_new, log_y_pred_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_log_train, annot=True, fmt=".1f")
plt.title('Logistic Regression Train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, log_y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[29]:


# Creates a confusion matrix
cm_log_test = confusion_matrix(y_test_new, log_y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_log_test, annot=True, fmt=".1f")
plt.title('Logistic Regression Test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, log_y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[30]:


#training 
TP_log_train = cm_log_train[1][1]
FP_log_train = cm_log_train[1][0]
FN_log_train = cm_log_train[0][1]
TN_log_train = cm_log_train[0][0]
print('TP is : {:.2f}.'.format(TP_log_train))
print('FP is : {:.2f}.'.format(FP_log_train))
print('FN is : {:.2f}.'.format(FN_log_train))
print('TN is : {:.2f}.'.format(TN_log_train))

#test
TP_log_test = cm_log_test[1][1]
FP_log_test = cm_log_test[1][0]
FN_log_test = cm_log_test[0][1]
TN_log_test = cm_log_test[0][0]
print('TP is : {:.2f}.'.format(TP_log_test))
print('FP is : {:.2f}.'.format(FP_log_test))
print('FN is : {:.2f}.'.format(FN_log_test))
print('TN is : {:.2f}.'.format(TN_log_test))


# In[31]:


#Now the normalize the diagonal entries
cm_log_train_full = cm_log_train.astype('float') / cm_log_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_log_train_full.diagonal()


# In[32]:


#Now the normalize the diagonal entries
cm_log_full_test = cm_log_test.astype('float') / cm_log_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_log_full_test.diagonal()


# In[33]:


plt.scatter(X_Test_new[log_y_pred_test==0, 0] , X_Test_new[log_y_pred_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[log_y_pred_test==1, 0] , X_Test_new[log_y_pred_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('Logistic Regression')
plt.show()


# ### Decision Trees

# In[34]:


#Decision Trees
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train_new, y_train_new) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
y_pred_dt_train = clf.predict(X_train_new)
y_pred_dt_test = clf.predict(X_Test_new)


# In[35]:


#calculate the scores
acc_train_dt = accuracy_score(y_train_new, y_pred_dt_train)
acc_test_dt = accuracy_score(y_test_new, y_pred_dt_test)
pre_train_dt = precision_score(y_train_new, y_pred_dt_train, average='macro')
pre_test_dt = precision_score(y_test_new, y_pred_dt_test, average='macro')
rec_train_dt = recall_score(y_train_new, y_pred_dt_train, average='macro')
rec_test_dt = recall_score(y_test_new, y_pred_dt_test, average='macro')
f1_train_dt = f1_score(y_train_new, y_pred_dt_train, average='macro')
f1_test_dt = f1_score(y_test_new, y_pred_dt_test, average='macro')


# In[36]:


print('Accuracy scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(acc_train_dt), 'and test: {:.2f}.'.format(acc_test_dt))
print('Precision scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(pre_train_dt), 'and test: {:.2f}.'.format(pre_test_dt))
print('Recall scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(rec_train_dt), 'and test: {:.2f}.'.format(rec_test_dt))
print('F1 scores of Decision Tree classifier are:',
      'train: {:.2f}'.format(f1_train_dt), 'and test: {:.2f}.'.format(f1_test_dt))


# In[37]:


from sklearn.metrics import classification_report
print('                DT Train set classification report')
print(classification_report(y_train_new, y_pred_dt_train))


# In[38]:


print('                DT Test set classification report')
print(classification_report(y_test_new, y_pred_dt_test))


# In[39]:


# Creates a confusion matrix
cm_dt_train = confusion_matrix(y_train_new, y_pred_dt_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_dt_train, annot=True, fmt=".1f")
plt.title('Decision Trees Train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, y_pred_dt_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[40]:


# Creates a confusion matrix
cm_dt_test = confusion_matrix(y_test_new, y_pred_dt_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_dt_test, annot=True, fmt=".1f")
plt.title('Decision Trees Test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, y_pred_dt_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[41]:


#training 
TP_dt_train = cm_dt_train[1][1]
FP_dt_train = cm_dt_train[1][0]
FN_dt_train = cm_dt_train[0][1]
TN_dt_train = cm_dt_train[0][0]
print('TP is : {:.2f}.'.format(TP_dt_train))
print('FP is : {:.2f}.'.format(FP_dt_train))
print('FN is : {:.2f}.'.format(FN_dt_train))
print('TN is : {:.2f}.'.format(TN_dt_train))

#test
TP_dt_test = cm_dt_test[1][1]
FP_dt_test = cm_dt_test[1][0]
FN_dt_test = cm_dt_test[0][1]
TN_dt_test = cm_dt_test[0][0]
print('TP is : {:.2f}.'.format(TP_dt_test))
print('FP is : {:.2f}.'.format(FP_dt_test))
print('FN is : {:.2f}.'.format(FN_dt_test))
print('TN is : {:.2f}.'.format(TN_dt_test))


# In[42]:


#Now the normalize the diagonal entries
cm_full_train = cm_dt_train.astype('float') / cm_dt_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_full_train.diagonal()


# In[43]:


#Now the normalize the diagonal entries
cm_full_test = cm_dt_test.astype('float') / cm_dt_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_full_test.diagonal()


# In[44]:


plt.scatter(X_Test_new[y_pred_dt_test==0, 0] , X_Test_new[y_pred_dt_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[y_pred_dt_test==1, 0] , X_Test_new[y_pred_dt_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('Decision Trees')
plt.show()


# ### k-Nearest Neighbors

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_new, y_train_new) #fit the model using the training data
#now check for both train and test data, how well the model learned the patterns
knn_y_pred_train = knn.predict(X_train_new)
knn_y_pred_test = knn.predict(X_Test_new)


# In[46]:


#calculate the scores
knn_acc_train_dt = accuracy_score(y_train_new, knn_y_pred_train)
knn_acc_test_dt = accuracy_score(y_test_new, knn_y_pred_test)
knn_pre_train_dt = precision_score(y_train_new, knn_y_pred_train, average='macro')
knn_pre_test_dt = precision_score(y_test_new, knn_y_pred_test, average='macro')
knn_rec_train_dt = recall_score(y_train_new, knn_y_pred_train, average='macro')
knn_rec_test_dt = recall_score(y_test_new, knn_y_pred_test, average='macro')
knn_f1_train_dt = f1_score(y_train_new, knn_y_pred_train, average='macro')
knn_f1_test_dt = f1_score(y_test_new, knn_y_pred_test, average='macro')


# In[47]:


print('Accuracy scores of k-Nearest Neighbors classifier are:',
      'train: {:.2f}'.format(knn_acc_train_dt), 'and test: {:.2f}.'.format(knn_acc_test_dt))
print('Precision scores of k-Nearest Neighbors classifier are:',
      'train: {:.2f}'.format(knn_pre_train_dt), 'and test: {:.2f}.'.format(knn_pre_test_dt))
print('Recall scores of k-Nearest Neighbors classifier are:',
      'train: {:.2f}'.format(knn_rec_train_dt), 'and test: {:.2f}.'.format(knn_rec_test_dt))
print('F1 scores of k-Nearest Neighbors classifier are:',
      'train: {:.2f}'.format(knn_f1_train_dt), 'and test: {:.2f}.'.format(knn_f1_test_dt))


# In[48]:


print('                KNN Test set classification report')
print(classification_report(y_test_new, knn_y_pred_test))


# In[49]:


print('                KNN Train set classification report')
print(classification_report(y_train_new, knn_y_pred_train))


# In[50]:


# Creates a confusion matrix
cm_knn_train = confusion_matrix(y_train_new, knn_y_pred_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_knn_train, annot=True, fmt=".1f")
plt.title('k-Nearest Neighbors Train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, knn_y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[51]:


# Creates a confusion matrix
cm_knn_test = confusion_matrix(y_test_new, knn_y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_knn_test, annot=True, fmt=".1f")
plt.title('k-Nearest Neighbors Test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, knn_y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[52]:


#training 
TP_knn_train = cm_knn_train[1][1]
FP_knn_train = cm_knn_train[1][0]
FN_knn_train = cm_knn_train[0][1]
TN_knn_train = cm_knn_train[0][0]
print('TP is : {:.2f}.'.format(TP_knn_train))
print('FP is : {:.2f}.'.format(FP_knn_train))
print('FN is : {:.2f}.'.format(FN_knn_train))
print('TN is : {:.2f}.'.format(TN_knn_train))

#test
TP_knn_test = cm_knn_test[1][1]
FP_knn_test = cm_knn_test[1][0]
FN_knn_test = cm_knn_test[0][1]
TN_knn_test = cm_knn_test[0][0]
print('TP is : {:.2f}.'.format(TP_knn_test))
print('FP is : {:.2f}.'.format(FP_knn_test))
print('FN is : {:.2f}.'.format(FN_knn_test))
print('TN is : {:.2f}.'.format(TN_knn_test))


# In[53]:


#Now the normalize the diagonal entries
cm_knn_train_full = cm_knn_train.astype('float') / cm_knn_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_knn_train_full.diagonal()


# In[54]:


#Now the normalize the diagonal entries
cm_knn_train_full = cm_knn_test.astype('float') / cm_knn_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_knn_train_full.diagonal()


# In[55]:


plt.scatter(X_Test_new[knn_y_pred_test==0, 0] , X_Test_new[knn_y_pred_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[knn_y_pred_test==1, 0] , X_Test_new[knn_y_pred_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('k-Nearest Neighbors')
plt.show()


# ### Naïve Bayes

# In[56]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_new, y_train_new) #fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
nb_y_pred_train = gnb.predict(X_train_new)
nb_y_pred_test = gnb.predict(X_Test_new)


# In[57]:


#calculate the scores
nb_acc_train = accuracy_score(y_train_new, nb_y_pred_train)
nb_acc_test = accuracy_score(y_test_new, nb_y_pred_test)
nb_pre_train = precision_score(y_train_new, nb_y_pred_train, average='macro')
nb_pre_test = precision_score(y_test_new, nb_y_pred_test, average='macro')
nb_rec_train = recall_score(y_train_new, nb_y_pred_train, average='macro')
nb_rec_test = recall_score(y_test_new, nb_y_pred_test, average='macro')
nb_f1_train = f1_score(y_train_new, nb_y_pred_train, average='macro')
nb_f1_test = f1_score(y_test_new, nb_y_pred_test, average='macro')


# In[58]:


print('Accuracy scores of Naïve Bayes classifier are:',
      'train: {:.2f}'.format(nb_acc_train), 'and test: {:.2f}.'.format(nb_acc_test))
print('Precision scores of Naïve Bayes classifier are:',
      'train: {:.2f}'.format(nb_pre_train), 'and test: {:.2f}.'.format(nb_pre_test))
print('Recall scores of Naïve Bayes classifier are:',
      'train: {:.2f}'.format(nb_rec_train), 'and test: {:.2f}.'.format(nb_rec_test))
print('F1 scores of Naïve Bayes classifier are:',
      'train: {:.2f}'.format(nb_f1_train), 'and test: {:.2f}.'.format(nb_f1_test))


# In[59]:


print('            Naïve Bayes Test set classification report')
print(classification_report(y_test_new, nb_y_pred_test))


# In[60]:


print('            Naïve Bayes Train set classification report')
print(classification_report(y_train_new, nb_y_pred_train))


# In[61]:


# Creates a confusion matrix
cm_nb_train = confusion_matrix(y_train_new, nb_y_pred_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_nb_train, annot=True, fmt=".1f")
plt.title('Naïve Bayes Train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, nb_y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[62]:


# Creates a confusion matrix
cm_nb_test = confusion_matrix(y_test_new, nb_y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_nb_test, annot=True, fmt=".1f")
plt.title('Naïve Bayes Test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, nb_y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[63]:


#training 
TP_nb_train = cm_nb_train[1][1]
FP_nb_train = cm_nb_train[1][0]
FN_nb_train = cm_nb_train[0][1]
TN_nb_train = cm_nb_train[0][0]
print('TP is : {:.2f}.'.format(TP_nb_train))
print('FP is : {:.2f}.'.format(FP_nb_train))
print('FN is : {:.2f}.'.format(FN_nb_train))
print('TN is : {:.2f}.'.format(TN_nb_train))

#test
TP_nb_test = cm_nb_test[1][1]
FP_nb_test = cm_nb_test[1][0]
FN_nb_test = cm_nb_test[0][1]
TN_nb_test = cm_nb_test[0][0]
print('TP is : {:.2f}.'.format(TP_nb_test))
print('FP is : {:.2f}.'.format(FP_nb_test))
print('FN is : {:.2f}.'.format(FN_nb_test))
print('TN is : {:.2f}.'.format(TN_nb_test))


# In[64]:


#Now the normalize the diagonal entries
cm_nb_train_full = cm_nb_train.astype('float') / cm_nb_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_nb_train_full.diagonal()


# In[65]:


#Now the normalize the diagonal entries
cm_nb_test_full = cm_nb_test.astype('float') / cm_nb_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_nb_test_full.diagonal()


# In[66]:


plt.scatter(X_Test_new[nb_y_pred_test==0, 0] , X_Test_new[nb_y_pred_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[nb_y_pred_test==1, 0] , X_Test_new[nb_y_pred_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('Naïve Bayes')
plt.show()


# In[67]:


#Support Vector Machines


# In[68]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_new, y_train_new) #fit the model using the training data
# now check for both train and test data, how well the model learned the patterns
svm_y_pred_train = svm.predict(X_train_new)
svm_y_pred_test = svm.predict(X_Test_new)


# In[69]:


#calculate the scores
svm_acc_train = accuracy_score(y_train_new, svm_y_pred_train)
svm_acc_test = accuracy_score(y_test_new, svm_y_pred_test)
svm_pre_train = precision_score(y_train_new, svm_y_pred_train, average='macro', zero_division=0)
svm_pre_test = precision_score(y_test_new, svm_y_pred_test, average='macro',zero_division=0)
svm_rec_train = recall_score(y_train_new, svm_y_pred_train, average='macro')
svm_rec_test = recall_score(y_test_new, svm_y_pred_test, average='macro')
svm_f1_train = f1_score(y_train_new, svm_y_pred_train, average='macro')
svm_f1_test = f1_score(y_test_new, svm_y_pred_test, average='macro')


# In[70]:


print('Accuracy scores of SVM classifier are:',
      'train: {:.2f}'.format(svm_acc_train), 'and test: {:.2f}.'.format(svm_acc_test))
print('Precision scores of SVM classifier are:',
      'train: {:.2f}'.format(svm_pre_train), 'and test: {:.2f}.'.format(svm_pre_test))
print('Recall scores of SVM classifier are:',
      'train: {:.2f}'.format(svm_rec_train), 'and test: {:.2f}.'.format(svm_rec_test))
print('F1 scores of SVM classifier are:',
      'train: {:.2f}'.format(svm_f1_train), 'and test: {:.2f}.'.format(svm_f1_test))


# In[71]:


print('                SVM Test set classification report')
print(classification_report(y_test_new, svm_y_pred_test))


# In[72]:


print('                SVM Train set classification report')
print(classification_report(y_train_new, svm_y_pred_train))


# In[73]:


# Creates a confusion matrix
cm_svm_test = confusion_matrix(y_test_new, svm_y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_svm_test, annot=True, fmt=".1f")
plt.title('SVM Test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, svm_y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[74]:


# Creates a confusion matrix
cm_svm_train = confusion_matrix(y_train_new, svm_y_pred_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_svm_train, annot=True, fmt=".1f")
plt.title('SVM Train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, svm_y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[75]:


#training 
TP_svm_train = cm_svm_train[1][1]
FP_svm_train = cm_svm_train[1][0]
FN_svm_train = cm_svm_train[0][1]
TN_svm_train = cm_svm_train[0][0]
print('TP is : {:.2f}.'.format(TP_svm_train))
print('FP is : {:.2f}.'.format(FP_svm_train))
print('FN is : {:.2f}.'.format(FN_svm_train))
print('TN is : {:.2f}.'.format(TN_svm_train))

#test
TP_svm_test = cm_svm_test[1][1]
FP_svm_test = cm_svm_test[1][0]
FN_svm_test = cm_svm_test[0][1]
TN_svm_test = cm_svm_test[0][0]
print('TP is : {:.2f}.'.format(TP_svm_test))
print('FP is : {:.2f}.'.format(FP_svm_test))
print('FN is : {:.2f}.'.format(FN_svm_test))
print('TN is : {:.2f}.'.format(TN_svm_test))


# In[76]:


#Now the normalize the diagonal entries
cm_svm_full = cm_svm_train.astype('float') / cm_svm_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_svm_full.diagonal()


# In[77]:


#Now the normalize the diagonal entries
cm_svm_full = cm_svm_test.astype('float') / cm_svm_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
cm_svm_full.diagonal()


# In[78]:


plt.scatter(X_Test_new[svm_y_pred_test==0, 0] , X_Test_new[svm_y_pred_test==0, 1],c='m',marker='o',s=20,label='Class 0' )
plt.scatter(X_Test_new[svm_y_pred_test==1, 0] , X_Test_new[svm_y_pred_test==1, 1],c='b',marker='x',label='Class 1' )
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend(loc='best')
plt.title('SVM')
plt.show()


# In[79]:


#Neural Networks


# In[80]:


# ANN
model =  Sequential()
model.add(Dense(units=8,activation='relu'))
model.add(Dropout(0.10))

model.add(Dense(units=4,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

# compile ANN
model.compile(loss='binary_crossentropy', optimizer='adam',metrics = ['accuracy'])

#Fit the model
# Train ANN
model.fit(x=X_train_new, 
          y=y_train_new, 
          epochs=120,
          validation_data=(X_Test_new, y_test_new), verbose=1
          )
# initilize the model
model.summary()


# In[81]:


# model history to df
loss_plot = pd.DataFrame(model.history.history)
accuracy_plot = pd.DataFrame(model.history.history)

#  accuracy and loss plot
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(14,4))
plt.style.use('seaborn')
ax1.plot(loss_plot.loc[:, ['loss']], label='Training loss');
ax1.plot(loss_plot.loc[:, ['val_loss']],label='Validation loss');
ax1.set_title('Training and Validation loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('Loss')
ax1.legend(loc="best");

ax2.plot(accuracy_plot.loc[:, ['accuracy']],label='Training_accuracy');
ax2.plot(accuracy_plot.loc[:, ['val_accuracy']], label='Validation_accuracy');
ax2.set_title('Training_and_Validation_accuracy');
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend(loc="best");


# In[82]:


y_pred_train = model.predict(X_train_new)
y_pred_test = model.predict(X_Test_new)


# In[83]:


y_pred_test = np.round(abs(y_pred_test))
y_pred_train = np.round(abs(y_pred_train))


# In[84]:


# now check for both train and test data, how well the model learned the patterns
ann_acc_train = accuracy_score(y_train_new, y_pred_train)
ann_acc_test = accuracy_score(y_test_new, y_pred_test)
ann_pre_train = precision_score(y_train_new, y_pred_train, average='macro')
ann_pre_test = precision_score(y_test_new, y_pred_test, average='macro')
ann_rec_train = recall_score(y_train_new, y_pred_train, average='macro')
ann_rec_test = recall_score(y_test_new, y_pred_test, average='macro')
ann_f1_train = f1_score(y_train_new, y_pred_train, average='macro')
ann_f1_test = f1_score(y_test_new, y_pred_test, average='macro')

#print the scores
print('Accuracy scores of ANN classifier are:',
      'train: {:.2f}'.format(ann_acc_train), 'and test: {:.2f}.'.format(ann_acc_test))
print('Precision scores of ANN classifier are:',
      'train: {:.2f}'.format(ann_pre_train), 'and test: {:.2f}.'.format(ann_pre_test))
print('Recall scores of ANN classifier are:',
      'train: {:.2f}'.format(ann_rec_train), 'and test: {:.2f}.'.format(ann_rec_test))
print('F1 scores of ANN classifier are:',
      'train: {:.2f}'.format(ann_f1_train), 'and test: {:.2f}.'.format(ann_f1_test))


# In[85]:


print(classification_report(y_test_new, np.round(abs(y_pred_test))))


# In[86]:


ann_acc_train


# In[102]:


# Creates a confusion matrix
ann_train = confusion_matrix(y_train_new, y_pred_train)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(ann_train, annot=True, fmt=".1f")
plt.title('ANN train set\nAccuracy:{0:.3f}'.format(accuracy_score(y_train_new, y_pred_train)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[104]:


# Creates a confusion matrix
ann_test = confusion_matrix(y_test_new, y_pred_test)
#plots the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(ann_test, annot=True, fmt=".1f")
plt.title('ANN test set\nAccuracy:{0:.3f}'.format(accuracy_score(y_test_new, y_pred_test)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[105]:


#training 
TP_ann_train = ann_train[1][1]
FP_ann_train = ann_train[1][0]
FN_ann_train = ann_train[0][1]
TN_ann_train = ann_train[0][0]
print('TP is : {:.2f}.'.format(TP_ann_train))
print('FP is : {:.2f}.'.format(FP_ann_train))
print('FN is : {:.2f}.'.format(FN_ann_train))
print('TN is : {:.2f}.'.format(TN_ann_train))

#test
TP_ann_test = ann_test[1][1]
FP_ann_test = ann_test[1][0]
FN_ann_test = ann_test[0][1]
TN_ann_test = ann_test[0][0]
print('TP is : {:.2f}.'.format(TP_ann_test))
print('FP is : {:.2f}.'.format(FP_ann_test))
print('FN is : {:.2f}.'.format(FN_ann_test))
print('TN is : {:.2f}.'.format(TN_ann_test))


# In[90]:


#Now the normalize the diagonal entries
ann_acc_train_full = ann_train.astype('float') / ann_train.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
ann_acc_train_full.diagonal()


# In[91]:


#Now the normalize the diagonal entries
ann_acc_test_full = ann_test.astype('float') / ann_test.sum(axis=1)[:, np.newaxis]

#The diagonal entries are the accuracies of each class
ann_acc_test_full.diagonal()


# In[92]:


training_samples = len(y_train_new)
training_samples_test = len(y_test_new)

results = pd.DataFrame({'Classifier_Name':['Linear Discriminant Analysis', 'Linear Discriminant Analysis', 'Logistic Regression', 'Logistic Regression', 'Decision Trees', 'Decision Trees', 'k-Nearest Neighbors', 'k-Nearest Neighbors', 'Naïve Bayes', 'Naïve Bayes', 'Support Vector Machines', 'Support Vector Machines', 'Neural Networks', 'Neural Networks' ],
    'Training_or_test_set':['Training set', 'Test set', 'Training set', 'Test set', 'Training set', 'Test set', 'Training set', 'Test set', 'Training set', 'Test set', 'Training set', 'Test set', 'Training set', 'Test set'],
    'Number_of_training_samples':[training_samples, training_samples_test, training_samples, training_samples_test, training_samples, training_samples_test, training_samples, training_samples_test, training_samples, training_samples_test, training_samples, training_samples_test, training_samples, training_samples_test],
    'Number_of_non_healthy_companies_in_training_sample': [non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test, non_healthy_training, non_healthy_test,],
                       'TP': [TP_lda_train, TP_lda_test, TP_log_train, TP_log_test, TP_dt_train, TP_dt_test, TP_knn_train, TP_knn_test, TP_nb_train, TP_nb_test, TP_svm_train, TP_svm_test, TP_ann_train, TP_ann_test], 
                       'TN': [TN_lda_train, TN_lda_test, TN_log_train, TN_log_test, TN_dt_train, TN_dt_test, TN_knn_train, TN_knn_test, TN_nb_train, TN_nb_test, TN_svm_train, TN_svm_test, TN_ann_train, TN_ann_test]  ,
                       'FP': [FP_lda_train, FP_lda_test, FP_log_train, FP_log_test, FP_dt_train, FP_dt_test, FP_knn_train, FP_knn_test, FP_nb_train, FP_nb_test, FP_svm_train, FP_svm_test, FP_ann_train, FP_ann_test],
                       'FN': [FN_lda_train, FN_lda_test, FN_log_train, FN_log_test, FN_dt_train, FN_dt_test, FN_knn_train, FN_knn_test, FN_nb_train, FN_nb_test, FN_svm_train, FN_svm_test, FN_ann_train, FN_ann_test],
                       'Precision': [pre_train_lda, pre_test_lda, pre_train_log, pre_test_log, pre_train_dt, pre_test_dt, knn_pre_train_dt, knn_pre_test_dt, nb_pre_train, nb_pre_test, svm_pre_train, svm_pre_test, ann_pre_train, ann_pre_test],
                       'Recall': [rec_train_lda, rec_test_lda, rec_train_log, rec_test_log, rec_train_dt, rec_test_dt, knn_rec_train_dt, knn_rec_test_dt, nb_rec_train, nb_rec_test, svm_rec_train, svm_rec_test, ann_rec_train, ann_rec_test],
                       'F1 score': [f1_train_lda, f1_test_lda, f1_train_log, f1_test_log, f1_train_dt, f1_test_dt, knn_f1_train_dt, knn_f1_test_dt, nb_f1_train, nb_f1_test, svm_f1_train, svm_f1_test, ann_f1_train, ann_f1_test],
                       'Accuracy': [acc_train_lda, acc_test_lda, acc_train_log, acc_test_log, acc_train_dt, acc_test_dt, knn_acc_train_dt, knn_acc_test_dt, nb_acc_train, nb_acc_test, svm_acc_train, svm_acc_test, ann_acc_train, ann_acc_test]
                      })
#create directory to store the output file
outdir = './OutputData'
if not os.path.exists(outdir):
    os.mkdir(outdir)
results.to_excel('OutputData\\Results.xlsx')


# In[93]:


results


# In[ ]:




