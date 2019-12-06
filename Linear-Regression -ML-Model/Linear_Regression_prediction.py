#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#company spend and profit information Given. Need to prdeict newly come spends profit 
df=pd.read_csv('50_Startups.csv')
print(df)


# In[3]:


#To daisplay top 5 values
df.head()


# In[4]:


#To display last 5 values
df.tail()


# In[5]:


df.head(1)


# In[6]:


df.info()


# In[7]:


df.describe()


# In[18]:


df=pd.read_csv('50_Startups.csv')


# In[20]:


# In ML only allow in numeric data.so, we changed string data to numeric one
df = pd.get_dummies(df,columns=['State'],drop_first = True)
df.head()


# In[28]:


#we have taken all rows and 6 columns
X = df.iloc[:,[0,1,2,3,4,5]]


# In[29]:


# y is column 3. profit column. y only going to predict.
Y=df.iloc[:,3]


# In[30]:


#sickit learn is an ML library for the python programming language 
from sklearn.model_selection import train_test_split


# In[31]:


#Train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[32]:


#Import Linear Regreesion
from sklearn.linear_model import LinearRegression


# In[33]:


lm =LinearRegression()


# In[35]:


#fitting x and y
lm.fit(X_train,y_train)


# In[36]:


y_pred = lm.predict(X_test)


# In[37]:


#History data prediction
y_test.head(1)


# In[38]:


y_pred[1]


# In[42]:


from sklearn.metrics import r2_score


# In[44]:


#Accurasy score
r2_score(y_pred,y_test)

