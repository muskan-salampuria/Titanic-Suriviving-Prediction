#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

#Reading Titanic file in CSV Format
df=pd.read_csv("F:\small_projects\Titanic_using_Naive_B\\titanic.csv")
target=df.Survived
df.head(3)


# In[2]:


#Dropping uneccessary columns
df.drop(['PassengerId','Name', 'SibSp', 'Parch','Cabin' ,'Ticket' ,'Embarked','Survived'],axis='columns',inplace=True)
df.head(5)


# In[3]:


#Fetures on which the surviving depends
inputs=df


# In[4]:


#converting Sex text column into binary numbers
dummies=pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[5]:


#merging dummy column to the input dataset
inputs=pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[6]:


#dropping text sex column
inputs.drop('Sex',axis='columns',inplace=True)
inputs.head(5)


# In[7]:


#checking which column has null values
inputs.columns[inputs.isna().any()]


# In[8]:


inputs.Age[:10]


# In[9]:


#assigning the mean value to all null values in the column Age
inputs.Age= inputs.Age.fillna(inputs.Age.mean())
inputs.head(10)


# In[10]:


#splitting test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(inputs,target,test_size=0.2)


# In[11]:


X_test.head(10)


# In[12]:


X_train.head(10)


# In[13]:


#GaussianNB Model
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()


# In[14]:


#Training the model
model.fit(X_train,y_train)


# In[15]:


#Calculating Score for the model
model.score(X_test,y_test)


# In[16]:


#Predicting surviving values for the test dataset
model.predict(X_test[:10])


# In[17]:


y_test[:10]


# In[18]:


#Calculating the Probability of not surviving or surviving
model.predict_proba(X_test[:10])


# In[ ]:


# If Predicted value is 1: Survived and  0: Not survived

