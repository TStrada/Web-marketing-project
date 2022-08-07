#!/usr/bin/env python
# coding: utf-8

# The RFM model provides a deterministic description of the value of each customer in term of the purchase behaviour. 
# The metrics chosen to describe the customer behaviours are:
# 1) RECENCY How recently does the customer purchase the last time?
# 2) FREQUENCY How often does the customer purchase?
# 3) MONETARY VALUE How much money does the customer spend?
# 
# Typically one express the RFM value either as a numerical scores (e.g. 1 to 10) or as a business-understandable 
# categories(e.g. Diamond, Gold, Silver, Bronze, …).
# 
# The first step in constructing the RFM categories is to refine the customer base perimeter by dividing customer in 
# inactive or lost, i.e. those customers who are not purchasing for a significantly long time, and active.
# The RFM category associated to inactive customer is simply “inactive”.
# To quantify what is meant for “a significantly long time”, it must defines a threshold:
# 
# PURCHASE TIME SCALE = 
# number of days such that the 80-90% of the customers repurchase within this time interval from last purchase.
# 
# One then construct the so called RF-matrix:
# The recency and frequency percentile groups are combined to define new classes describing the customers loyalty status.
# 
# Finally, the RFM-classes are obtained by combining the RF-classeswith the Monetary Value groups.
# 

# In[1]:


import pandas as pd
import numpy as np
import re
import os
from scipy import stats
from datetime import timedelta
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import squarify


# ### EDA (Explorative Data Analysis )

# In[139]:


df = pd.read_csv('raw_7_tic.csv', sep = ';', 
                decimal = ',')


# In[140]:


# print(df.dtypes)
## Transform 'DATETIME' to datetime format.
df['DATETIME'] = pd.to_datetime(df['DATETIME'])
print(df.dtypes)


# In[141]:


print(len(df))
df.head(5)


# Features: 
# 1) ID_SCONTRINO = Codice identificativo dell'acquisto
# 2) ID_CLI = Codice identificativo del cliente
# 3) ID_NEG = Codice Identificativo del negozio
# 4) COD_REPARTO = Codice del reparto
# 5) DIREZIONE = 
# 6) IMPORTO_LORDO = Prezzo base
# 7) SCONTO = sconto applicato (euro)
# 8) DATETIME = data acquisto

# In[142]:


# Missing values
missing_values_count = df.isnull().sum()
print('N° missing values: ', missing_values_count[missing_values_count > 0])
## There are no missing values


##---------------------------------------------------------------------------------------------

print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))
print('{:,} transactions don\'t have a customer id'
      .format(df[df.ID_CLI.isnull()].shape[0]))
print('Transactions timeframe from {} to {}'.format(df['DATETIME'].min(),
                                    df['DATETIME'].max()))

## All transactions have a customer. The dataset covers 1 year purchases .
##--------------------------------------------------------------------------------------------

## Check duplicates purchases.
print('{:,} duplicated purchases'
      .format(len(df[df.duplicated(subset=['ID_SCONTRINO', 'ID_CLI',
                                       'ID_NEG', 'ID_ARTICOLO', 'COD_REPARTO',
                                       'DIREZIONE', 'DATETIME']
                              )
                    ]
                 )
             )
     )

##They may have different 'IMPORTO_LORDO' and 'SCONTO', so we keep the row that specify the 'SCONTO' amount. 
df.drop_duplicates(subset=['ID_SCONTRINO', 'ID_CLI', 'ID_NEG', 'ID_ARTICOLO', 'COD_REPARTO',
       'DIREZIONE', 'DATETIME'], inplace = True)

print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))

df.head(25)

##--------------------------------------------------------------------------------------------------------
## Check if there are negative 'IMPORT_LORDO' values
negative = df[df['IMPORTO_LORDO'] < 0]
print('Negative values: ' + str(len(negative)))

## Drop negative values
df = df[df['IMPORTO_LORDO'] >= 0]
print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))


# ### Recency = snapshot_date - last purchase
# ### Frequency = count(Id_scontrino)
# ### Monetary Value = IMPORTO_LORDO - SCONTO

# In[143]:


print('Last purchase: ' + str(df.DATETIME.max()))
print('First purchase: ' + str(df.DATETIME.min()))


# In[144]:


## Set last date as point cut to calculate the day next purchase 
max_date = df.DATETIME.max()

next_day_purchase = df.groupby(['ID_CLI']).agg({
    'DATETIME': lambda x: (max_date - x.max()).days})


# In[145]:


print('Summary: ', next_day_purchase.DATETIME.describe())
print('80% ', np.percentile(a = next_day_purchase.DATETIME, q = 80))
print('90% ', np.percentile(a = next_day_purchase.DATETIME, q = 90))


# ###### Mean is 130 days while median is 106 days. 
# ###### So we should consider as inactive the purchases over 234 days from the last purchase,
# ###### but in agreement with common use the cut point is set at 60 days before the last purchase.
# ###### So the cut point date is 2019-02-28.

# In[146]:


## Days Next Purchase Curve
next_day_purchase.reset_index(inplace=True)
data = (next_day_purchase.groupby('DATETIME').count() / len(next_day_purchase))*100 
data.reset_index(inplace = True)
np.cumsum(data['ID_CLI'])

plt.plot(data['DATETIME'], np.cumsum(data['ID_CLI']))


# In[147]:


import datetime
## Split dataset into active and inactrive
df_active = df[df['DATETIME'] > datetime.datetime.strptime('2019-02-28', '%Y-%m-%d')]
df_active.head(100)


# In[148]:


## Create snapshot date to calculate the recency.
## The date difference will give us how recent the last transaction is.
snapshot_date = df_active['DATETIME'].max() + timedelta(days=1)
print(snapshot_date)

## Calculate total price = IMPORTO_LORDO - SCONTO
df_active['Total Price'] = df_active['IMPORTO_LORDO'] - df_active['SCONTO']

## Grouping by CustomerID
## Recency = last day available - last purchase day for customers
dt = df_active.groupby(['ID_CLI']).agg({
        'DATETIME': lambda x: (snapshot_date - x.max()).days,
        'ID_SCONTRINO': 'count',
        'Total Price': 'sum'})

# Rename the columns 
dt.rename(columns={'DATETIME': 'Recency',
                   'ID_SCONTRINO': 'Frequency',
                   'Total Price': 'MonetaryValue'}, inplace=True)


# ## Plot RFM distribution

# In[150]:


plt.figure(figsize=(12,10))

# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(dt['Recency'])

# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(dt['Frequency'])

# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(dt['MonetaryValue'])

# Show the plot
plt.show()


# ### Calculating R, F and M groups

# In[160]:


# Create labels for Recency and Frequency
r_labels = range(4, 0, -1); f_labels = range(1, 5)

# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(dt['Recency'], q=4, labels=r_labels)

# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(dt['Frequency'], q=4, labels=f_labels)

# Create labels for MonetaryValue
m_labels = range(1, 5)

# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(dt['MonetaryValue'], q=4, labels=m_labels)


# Create new columns R and F 
DT_PROCESS = dt.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)

## Assign 'Low', 'Medium', 'High' to each RFM values
def class_transform(x):
    if x == 1:
        return 'Low'
    if (x > 1) & (x <= 3):
        return 'Medium'
    if x == 4:
        return  'High'
        
DT_PROCESS['R Classes'] = DT_PROCESS.R.apply(lambda x:class_transform(x)) 
DT_PROCESS['F Classes'] = DT_PROCESS.F.apply(lambda x:class_transform(x)) 
DT_PROCESS['M Classes'] = DT_PROCESS.M.apply(lambda x:class_transform(x)) 
DT_PROCESS


# In[162]:


print('Recency summary: ')
print(DT_PROCESS['R Classes'].describe())
print('-------------------------')

print('Frequency summary: ')
print(DT_PROCESS['F Classes'].describe())
print('-------------------------')

print('Monetary Value summary: ')
print(DT_PROCESS['M Classes'].describe())
print('-------------------------')

## The most frequent class is 'Medium' for RFM values. (Notice that it's the largest class)
## More or less 50% of customers have bought within 1 month


# In[163]:


# Concating the RFM quartile values to create RFM Segments
def join_rfm(x): return (str(x['R']) + str(x['F']) + str(x['M']))

DT_PROCESS['RFM_Segment_Concat'] = DT_PROCESS.apply(join_rfm, axis=1)
DT_RFM = DT_PROCESS
DT_RFM.head()


# In[164]:


# Count num of unique segments
DT_RFM_unique = DT_RFM.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
print('N° of unique segments:', DT_RFM_unique.sum())

# Calculate RFM_Score
DT_RFM['RFM_Score'] = DT_RFM[['R','F','M']].sum(axis=1)
print('--------------------------')
print(DT_RFM['RFM_Score'].head())


# In[165]:


# Define FMCG_rfm_level function in order to define the RFM classes
def FMCG_rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Diamond'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Gold'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Silver'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Bronze'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Copper'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Tin'
    else:
        return 'Cheap'

# Create a new variable RFM_Level
DT_RFM['RFM_Level'] = DT_RFM.apply(FMCG_rfm_level, axis=1)
DT_RFM.head(15)


# In[202]:


## Calculate average values for each RFM_Level, and return a size of each segment
DT_RFM_AGG = DT_RFM.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'RFM_Level': 'count'
}).round(1)

# Print the aggregated dataset
DT_RFM_AGG = DT_RFM_AGG.rename(columns={'RFM_Level': 'Count'})
print(DT_RFM_AGG.sort_values(by='Count'))

## The most frequent class is Diamond which buy within 17 days on average.
## They spend 538.5 on average.
## The other two most frequent class are Bronze and Silver 


# In[203]:


DT_RFM_AGG.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']

#Create our plot and resize it.
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
squarify.plot(sizes=DT_RFM_AGG['Count'], 
              label=['Diamond',
                     'Bronze',
                     'Silver',
                     'Gold',
                     'Copper', 
                     'Tin', 
                     'Cheap'], alpha=.6 )

plt.title("RFM Segments",fontsize=18,fontweight="bold")
plt.axis('off')
plt.show()

