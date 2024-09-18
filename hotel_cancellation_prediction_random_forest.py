#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('hotel_bookings.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# # 2)	Print the unique values in all columns

# In[5]:


for col in df.columns:
    print(col,df[col].unique())


# # 3)	Fill nan value in country with ‘other’

# In[6]:


df['country'] = df['country'].replace(np.nan, 'Others')


# In[7]:


df['country'].unique()


# In[8]:


df.isnull().sum()


# In[9]:


df['agent'].unique()


# # 4)	Fill nan in agent with mean of agent columns

# In[10]:


df['agent'] = df['agent'].replace(np.nan, df['agent'].mean())


# In[11]:


df.isnull().sum()


# In[12]:


df['company'] = df['company'].replace(np.nan, df['company'].mean())


# In[13]:


df.isnull().sum()


# # 5)	Drop all the remaining null values

# In[14]:


df = df.dropna()


# In[15]:


df.isnull().any()


# # 6)	Plot the count of adult and children with help of a  bar plot

# In[16]:


df['adults'].value_counts().plot(kind = 'bar')


# In[17]:


df['children'].value_counts().plot(kind = 'bar')


# In[18]:


df.info()


# In[19]:


df = df.drop('reservation_status_date' , axis = 1 )                                        


# In[20]:


df.columns


# # 7)	Perform Label encoding on categorical columns

# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


enc = LabelEncoder()


# In[23]:


df['hotel'] = enc.fit_transform(df['hotel'])


# In[24]:


df['country'] = enc.fit_transform(df['country'])
df['hotel'] = enc.fit_transform(df['hotel'])
df['market_segment'] = enc.fit_transform(df['market_segment'])
df['distribution_channel'] = enc.fit_transform(df['distribution_channel'])
df['meal'] = enc.fit_transform(df['meal'])
df['reserved_room_type'] = enc.fit_transform(df['reserved_room_type'])
df['assigned_room_type'] = enc.fit_transform(df['assigned_room_type'])
df['deposit_type'] = enc.fit_transform(df['deposit_type'])
df['customer_type'] = enc.fit_transform(df['customer_type'])
df['reservation_status'] = enc.fit_transform(df['reservation_status'])


# In[25]:


df.info()


# In[26]:


df['arrival_date_month'] = enc.fit_transform(df['arrival_date_month'])


# In[27]:


df.info()


# # model Building

# In[28]:


from sklearn.model_selection import train_test_split


# # 1.	Create features and target data

# In[29]:


X = df.drop('is_canceled',axis=1)
y = df.is_canceled


# # 2.	Split into training & testing

# In[30]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)


# In[31]:


X_train.shape


# In[32]:


X_test.shape


# # 3.	Apply Random forest classifier on data

# In[33]:


from sklearn.ensemble import RandomForestClassifier

params_rf = {'max_depth': 16,
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             'n_estimators': 100,
             'random_state': 12345}

model_rf = RandomForestClassifier(**params_rf)


# In[34]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# # 4.	Create function which show Precision score, recall score, accuracy, classification report and confusion matrix.

# In[35]:


def run_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train,y_train.ravel())
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("pricison_score: ",precision_score(y_test, y_pred))
    print("recall_score: ",recall_score(y_test, y_pred))
    print("Accuracy = {}".format(accuracy))
    print(classification_report(y_test,y_pred,digits=5))
    print(confusion_matrix(y_test,y_pred))
    


# In[ ]:





# In[36]:


run_model(model_rf,X_train, y_train, X_test, y_test)


# In[ ]:





# In[ ]:




