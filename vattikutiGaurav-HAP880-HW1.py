#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


# In[2]:


root = r'C:\Users\vatti\OneDrive\Documents\Classes\HAP 880\Week_1'
os.chdir(root)

df = pd.read_csv('highUtilizationPredictionV2wco.csv')


# In[3]:


df.isnull().values.any()


# In[4]:


cols = df.columns.tolist()


# In[5]:


x = pd.DataFrame(df[(df['race']=='B') & (df['age']== 70)])


# In[6]:


cols_remove = ['race', 'patient_id', 'HighUtilizationY2', 'claimCount']


# In[7]:


X = df[list(set(cols).difference(set(cols_remove)))]


# In[8]:


y = df['HighUtilizationY2']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.33, random_state=42)


# In[10]:


lr = LogisticRegression()

lr_train = lr.fit(X_train, y_train)
lr_train.coef_


# In[11]:


lr_test = lr.fit (X_test, y_test)
lr_test.coef_


# In[12]:


prob_train = lr_train.predict_proba (X_train)[:, 1]
prob_train


# In[13]:


prob_test = lr_test.predict_proba (X_test)[:, 1]
prob_test


# In[14]:


fpr_train, tpr_train, thresholds_train = roc_curve(y_train, prob_train)


# In[15]:


thresholds_train


# In[16]:


auc_train = auc(fpr_train, tpr_train)
auc_train


# In[17]:


fpr_test, tpr_test, thresholds_test = roc_curve(y_test, prob_test)


# In[18]:


thresholds_test


# In[19]:


auc_test = auc(fpr_test, tpr_test)
auc_test


# In[20]:


plt.plot(fpr_train, tpr_train, label="Train_Data, AUC="+str(auc_train), color = 'c')

plt.plot(fpr_test, tpr_test, label="Test_Data, AUC="+str(auc_test), color = 'm')

plt.show()

