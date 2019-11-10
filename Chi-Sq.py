
# coding: utf-8

# In[56]:

# Import modules
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sb

# Configure inline mode
get_ipython().magic('matplotlib inline')


# In[57]:

dat = pd.read_csv("cleaned_df.txt")


# In[58]:

dat.head()


# In[59]:

dat.info()


# In[60]:

df = pd.DataFrame(dat)


# In[61]:

import scipy as sp


# In[62]:

from scipy.stats import chi2_contingency
from scipy.stats import chi2


# In[63]:

df3 = df[["MV303", "MV301"]]


# In[64]:

df3.head(3)


# In[65]:

crosstab = pd.crosstab(df["MV303"],df["MV301"])


# In[67]:

sp.stats.chi2_contingency(crosstab)


# In[68]:

chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab)


# In[69]:

p


# In[70]:

crosstab2 = pd.crosstab(df["MV303"],df["MV302"])


# In[71]:

chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab2)


# In[72]:

p


# In[73]:

crosstab3 = pd.crosstab(df["MV301"],df["MV302"])


# In[74]:

chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab3)


# In[75]:

p


# In[76]:

crosstab4 = pd.crosstab(df["MV303"],df["MV304"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab4)
p


# In[77]:

crosstab5 = pd.crosstab(df["MV303"],df["MV101"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab5)
p


# In[78]:

crosstab6 = pd.crosstab(df["MV303"],df["MV201"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab6)
p


# In[79]:

crosstab7 = pd.crosstab(df["UV401"],df["P401"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab7)
p


# In[80]:

crosstab8 = pd.crosstab(df["MV303"],df["is_attack"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab8)
p


# In[81]:

crosstab9 = pd.crosstab(df["UV401"],df["is_attack"])
chi2, p, dof, expected = sp.stats.chi2_contingency(crosstab9)
p


# In[ ]:



