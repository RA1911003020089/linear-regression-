#!/usr/bin/env python
# coding: utf-8

# # import libraries

# In[93]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[94]:


import warnings
warnings.filterwarnings('ignore')


# In[95]:


housing = pd.DataFrame(pd.read_csv("Housing.csv"))


# In[96]:


housing


# In[5]:


housing.head()


# In[6]:


housing.info()


# In[7]:


housing.describe()


# # data cleaning

# In[8]:


housing.isnull().sum()*100/housing.shape[0]


# # exploring data

# In[9]:


sns.pairplot(housing)
plt.show()


# In[10]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[29]:


scaler = MinMaxScaler()


# In[31]:


df_train.head()


# In[32]:


df_train.describe()


# In[63]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# In[64]:


y_train = df_train.pop('price')
X_train = df_train


# In[65]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[69]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[70]:


rfe = RFE(lm, 6)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[71]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[72]:


col = X_train.columns[rfe.support_]
col


# In[73]:


X_train.columns[~rfe.support_]


# In[74]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[75]:


# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)


# In[ ]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[ ]:


#Let's see the summary of our linear model
print(lm.summary())


# In[76]:


# Calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[77]:


vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[78]:


y_train_price = lm.predict(X_train_rfe)


# In[79]:


res = (y_train_price - y_train)


# In[80]:


# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# In[82]:


plt.scatter(y_train,res)
plt.show()


# In[83]:


# There may be some relation in the error terms.


# In[84]:


num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']


# In[85]:


df_test[num_vars] = scaler.fit_transform(df_test[num_vars])


# In[86]:


y_test = df_test.pop('price')
X_test = df_test


# In[87]:


# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)


# In[88]:


# Now let's use our model to make predictions.


# In[89]:


# Creating X_test_new dataframe by dropping variables from X_test
X_test_rfe = X_test[X_train_rfe.columns]


# In[90]:


# Making predictions
y_pred = lm.predict(X_test_rfe)


# In[91]:


from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)


# In[92]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[ ]:




