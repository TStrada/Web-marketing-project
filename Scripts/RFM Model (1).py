#!/usr/bin/env python
# coding: utf-8

# The RFM model provides a deterministic description of the value of each customer in term of the purchase behaviour. 
# The metrics chosen to describe the customer behaviours are:
# 1) RECENCY: How recently does the customer purchase the last time?
# 2) FREQUENCY: How often does the customer purchase?
# 3) MONETARY VALUE: How much money does the customer spend?
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
import datetime


# ### EDA (Explorative Data Analysis )

# In[35]:


df = pd.read_csv('raw_7_tic.csv', sep = ';', 
                decimal = ',', encoding = 'latin-1')
# a = df.groupby('ID_CLI').count().rename(columns={'DATETIME': 'Count'})['Count']
# df = pd.merge(df, a, on='ID_CLI')


# In[36]:


# print(df.dtypes)
## Transform 'DATETIME' into datetime format.
## Transform 'COD_REPARTO' into Category format.
## Transform 'DIREZIONE' into Category format.

df['DATETIME'] = pd.to_datetime(df['DATETIME'])
df[['COD_REPARTO', 'DIREZIONE']] = df[['COD_REPARTO', 'DIREZIONE']].apply(lambda x: x.astype('category'))
df.dtypes


# In[37]:


print(len(df))
df.head(5)


# Features: 
# 1) ID_SCONTRINO = Purchase tracking number
# 2) ID_CLI = Customer tracking number
# 3) ID_NEG = Shop tracking number
# 4) COD_REPARTO = Unit tracking number
# 5) DIREZIONE = purchase (+1) or return (-1)
# 6) IMPORTO_LORDO = Gross amount
# 7) SCONTO = discount
# 8) DATETIME = Purchase date

# In[38]:


## Split 'DATETIME' into 'DATE' and 'TIME'
df['DATE'] = df['DATETIME'].dt.date
df['TIME'] = df['DATETIME'].dt.time
df.head()


# In[39]:


# Missing values
missing_values_count = df.isnull().sum()
print('N° missing values: ', missing_values_count[missing_values_count > 0])
print(' ')
## There are no missing values


##---------------------------------------------------------------------------------------------

print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))
print('{:,} transactions don\'t have a customer id'
      .format(df[df.ID_CLI.isnull()].shape[0]))
print('Transactions timeframe from {} to {}'.format(df['DATE'].min(),
                                    df['DATE'].max()))
print(' ')

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
print(' ')

##They may have different 'IMPORTO_LORDO' and 'SCONTO', so we keep the row that specify the 'SCONTO' amount. 
df.drop_duplicates(subset=['ID_SCONTRINO', 'ID_CLI', 'ID_NEG', 'ID_ARTICOLO', 'COD_REPARTO',
       'DIREZIONE', 'DATETIME'], inplace = True)

print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))
print(' ')

df.head(25)

##--------------------------------------------------------------------------------------------------------
## Check if there are negative 'IMPORT_LORDO' values
negative = df[df['IMPORTO_LORDO'] < 0]
print('Negative values: ' + str(len(negative)))
## Check if negative values are associated with negative DIREZIONE
print('Number DIREZIONE = -1 link to negative values:' + str(len(negative[negative['DIREZIONE'] == -1])))
print('Number DIREZIONE = 1 link to negative values:' + str(len(negative[negative['DIREZIONE'] == 1])))
print(' ')
## There are 563 mismatches between 'DIREZIONE' and 'IMPORTO_LORDO'. They could be mistakes.  

## Drop negative values
df = df[df['IMPORTO_LORDO'] >= 0]
print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))

##-------------------------------------------------------------------------

## Calculate IMPORTO_NETTO = IMPORTO_LORDO - SCONTO
df['IMPORTO_NETTO'] = df['IMPORTO_LORDO'] - df['SCONTO']

## Drop purchases with IMPORTO_NETTO = 0
zero = df[df['IMPORTO_NETTO'] == 0]
print('N° gratis purchases: {}'.format(len(zero)))
df = df[df['IMPORTO_NETTO'] > 0] 

print('{:,} rows; {:,} columns'
      .format(df.shape[0], df.shape[1]))


# In[7]:


plt.figure(figsize=(12,10))

# Plot COD_REPARTO: n° purchases
cod_reparto_list_count = df.groupby(['COD_REPARTO']).count().rename(columns={'DATETIME': 'Count'})['Count']
plt.subplot(2, 1, 2); cod_reparto_list_count.plot(kind="bar", align='center')

# Plot average IMPORTO_LORDO and SCONTO by COD_REPARTO
cod_reparto_avg = df.groupby(['COD_REPARTO'], as_index = False).mean()[['IMPORTO_LORDO',
                                                      'SCONTO']].rename(columns={'IMPORTO_LORDO': 'IMPORTO_LORD_AVG', 
                                                                                 'SCONTO': 'SCONTO_AVG'})
plt.subplot(2, 1, 2); cod_reparto_avg.plot(kind="bar", align='center')
plt.yscale('log')

plt.show()

## Department n°3 sells more products and keeps IMPORTO_LORDO and SCONTO on mean.
## Department n°5 keeps highest IMPORTO_LORDO and SCONTO but is the worst products seller.


# In[8]:


## Upload the dataset containing customers informations
df_1 = pd.read_csv('raw_1_cli_fid.csv', sep = ';', 
                decimal = ',', encoding = 'latin-1')
df_1.head(5)


# In[9]:


## Check which customers have fidelity informations
print('N° customers from df_1 in df: ' +
      str(df_1['ID_CLI'].isin(df['ID_CLI']).value_counts().get(True)))
print('N° customers from df_1 in df(%): ' + 
      str((df_1['ID_CLI'].isin(df['ID_CLI']).value_counts(normalize = True).get(True)*100).round(2)) + '%')
print('')

print('N° customers from df in df_1: ' + 
      str(df_1['ID_CLI'].isin(df['ID_CLI']).value_counts().get(True)))
print('N° customers from df in df_1(%): ' + 
      str((df['ID_CLI'].isin(df_1['ID_CLI']).value_counts(normalize = True).get(True)*100).round(2)) + '%')

## Not all customers have purchases informations.


# In[10]:


## Fidelity type about customers with purchases informations
merge_df_df1 = pd.merge(df, df_1, on = 'ID_CLI').drop('ID_NEG_x', axis = 1).rename(columns={'ID_NEG_y': 'ID_NEG'})
merge_df_df1.reset_index(inplace = True, drop = True)

## Plot
merge_df_df1.groupby('COD_FID', as_index=False).agg(Count = ('DATETIME', 'count')).plot.bar(x='COD_FID', 
                                                                                            y='Count', 
                                                                                            rot=0,
                                                                                            )

## Customers often have standard or premium class fidelity


# ### RECENCY = days passed from last purchase for each customer
# First of all we set a cut point in order to divide customer base into active and inactive.
# The cut point is choosen considering the next purchase curve

# In[11]:


df_next_day = df.groupby('ID_CLI', as_index = False).agg(PRIMO_ACQUISTO = ('DATE', 'min'), 
                                                         ULTIMO_ACQUISTO = ('DATE', 'max'),
                                                         NUMERO_ACQUISTI = ('DATE', 'nunique'))
df_next_day.head(5)


# In[12]:


df_7_next = df.sort_values(by = 'DATE')
df_7_next.reset_index(inplace = True, drop = True)
df7_day_next = df_7_next.groupby(['ID_CLI', 'DATE']).agg(PRIMO_ACQUISTO = ('DATE', 'min'))
df7_day_next.reset_index(inplace = True)
df7_day_next.drop(['PRIMO_ACQUISTO'], axis = 1, inplace = True)
df7_day_next = df7_day_next.groupby('ID_CLI')[['DATE']].nth(-2)
df7_day_next.reset_index(inplace = True)
df7_day_next.rename(columns = {'DATE' : 'PENULTIMO_ACQUISTO'}, inplace = True)
df7_day_next


# In[13]:


df7_day_next[df7_day_next['ID_CLI'] == 18]


# In[14]:


df7_day_next_purchase = pd.merge(df_next_day, df7_day_next, on = 'ID_CLI', how = 'left')

df7_day_next_purchase['NEXT_PURCHASE_DAY'] = (df7_day_next_purchase['ULTIMO_ACQUISTO'] - df7_day_next_purchase['PENULTIMO_ACQUISTO']).dt.days
df7_day_next_purchase['NEXT_PURCHASE_DAY'] = df7_day_next_purchase['NEXT_PURCHASE_DAY'].fillna(0)
# df7_day_next_purchase = df7_day_next_purchase[df7_day_next_purchase['NEXT_PURCHASE_DAY'] != 0]
df7_day_next_purchase


# In[15]:


sns.kdeplot(data = df7_day_next_purchase, x = 'NEXT_PURCHASE_DAY', cumulative = True)


# In[16]:


print('Last purchase: ' + str(df.DATE.max()))
print('First purchase: ' + str(df.DATE.min()))


# In[17]:


## Set last date as point cut to calculate the day next purchase 
max_date = df.DATETIME.max()

next_day_purchase = df.groupby(['ID_CLI']).agg({
    'DATETIME': lambda x: (max_date - x.max()).days})


# In[18]:


print('Summary: ', next_day_purchase.DATETIME.describe())
print('80% ', np.percentile(a = next_day_purchase.DATETIME, q = 80))
print('90% ', np.percentile(a = next_day_purchase.DATETIME, q = 90))


# In[19]:


## Days Next Purchase Curve
data = (next_day_purchase.groupby('DATETIME').count() / len(next_day_purchase))*100 
data.reset_index(inplace = True)
np.cumsum(data['ID_CLI'])

plt.plot(data['DATETIME'], np.cumsum(data['ID_CLI']))
plt.xlabel('Average days next purchase')
plt.ylabel('Customer base %')
plt.axhline(y=80, color='y', linestyle='-')
plt.vlines(x = 234, ymin = 0, ymax = 100 , color='r', linestyle='-')


# Mean is 130 days while median is 106 days. 
# The 80% of customers buy the next product within 234 days.
# So we should consider as inactive the purchases over 234 days from the last purchase,
# but in agreement with common use the cut point is set at 60 days before the last purchase.
# So the cut point date is 2019-02-28.

# In[ ]:


## Split dataset into active and inactive
df_active = df[df['DATETIME'] > datetime.datetime.strptime('2019-02-28', '%Y-%m-%d')]
df_active.head(50)


# In[ ]:


## Create snapshot date to calculate the recency.
## The date difference will give us how recent the last transaction is.
snapshot_date = df_active['DATETIME'].max()
print('snapshot_date = ' + str(snapshot_date))

## Grouping by CustomerID
## Frequency = count(Id_scontrino)
## Recency = last day available - last purchase day for customers
## Monetary value = IMPORTO_LORDO - SCONTO

dt = df_active.groupby(['ID_CLI']).agg({
        'DATETIME': lambda x: (snapshot_date - x.max()).days,
        'ID_SCONTRINO': 'count',
        'Total Price': 'sum'})

## Rename the columns 
dt.rename(columns={'DATETIME': 'Recency',
                   'ID_SCONTRINO': 'Frequency',
                   'IMPORTO_NETTO': 'MonetaryValue'}, inplace=True)


# ## Plot RFM distribution

# In[ ]:


plt.figure(figsize=(12,10))

# Plot distribution of R
plt.subplot(3, 1, 1); sns.distplot(dt['Recency'])

# Plot distribution of F
plt.subplot(3, 1, 2); sns.distplot(dt['Frequency'])

# Plot distribution of M
plt.subplot(3, 1, 3); sns.distplot(dt['MonetaryValue'])

# Show the plot
plt.show()

##


# ### Calculating R, F and M groups

# In[ ]:


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


# In[ ]:


print('Recency summary: ')
print(DT_PROCESS['R Classes'].describe())
print('-------------------------')

print('Frequency summary: ')
print(DT_PROCESS['F Classes'].describe())
print('-------------------------')

print('Monetary Value summary: ')
print(DT_PROCESS['M Classes'].describe())
print('-------------------------')

## The most frequent class is 'Medium' for R, F and M values. (Notice that it's the largest class)
## More or less 50% of customers have bought within 1 month


# In[ ]:


## Plot
# hue = DT_PROCESS['R Classes'].unique()
# dt_plot_class = DT_PROCESS[['R Classes', 'F Classes', 'M Classes']]
# # plt.plot(dt_plot_class, dt_plot_class.count_values())
# dt_plot_class.value_counts()


# In[ ]:


## Create new label for R and F classes
R_F = DT_PROCESS[['R Classes', 'F Classes']]

def join_rf_class(x): return (str(x['R Classes']) + '-' +  str(x['F Classes']))

R_F['Combine Class'] = DT_PROCESS.apply(join_rf_class, axis=1)

diz_label = {'Medium-High': 'Top', 
             'Low-Medium': 'Engaged', 
             'High-High': 'Leaving Top',
             'Low-High': 'Top',
             'Low-Low': 'One Timer',
             'Medium-Medium': 'Engaged',
             'High-Medium': 'Leaving',
             'Medium-Low': 'One Timer', 
             'High-Low': 'Leaving'}

R_F['R-F Class'] = R_F['Combine Class'].apply(lambda x: diz_label[x])
R_F.head(5)


# In[ ]:


## Count R-F class
print('Most frequent class: ')
print(R_F['R-F Class'].describe()[['top', 'freq']])

## Percentage
print(' ')
print('Class percentage (%): ')
percentage = (R_F['R-F Class'].value_counts(normalize = True)*100).round(2)
print(str((R_F['R-F Class'].value_counts(normalize = True)*100).round(2)))

## Plot
percentage.plot.bar()
# plot.xlabel('R-F Class')

## Class definition:
## One-Timer: customers who don't have bought recently and not very frequently
#clienti che hanno acquistato recentemente/abbastanza recentementente ma con scarsa frequenza
## Leaving: 
#clienti che non hanno acquistato recentemente e con scarsa/media frequenza
## Engaged:
# clienti che hanno acquistato recentemente/abbastanza recentemente e con media frequenza
## Top:
# clienti che hanno acquistato recentemente/abbastanza recentemente e con alta frequenza
## Leaving Top:
# clienti che non hanno acquistato di recente ma con alta frequenza
## The most part of customers are classified as Engaged and then as One Timer
## gran parte dei clienti vengono classificati come clienti top


# In[ ]:


# Concating the RFM quartile values to create RFM Segments
def join_rfm(x): return (str(x['R']) + str(x['F']) + str(x['M']))

DT_PROCESS['RFM_Segment_Concat'] = DT_PROCESS.apply(join_rfm, axis=1)
DT_RFM = DT_PROCESS
DT_RFM.head()


# In[ ]:


# Count num of unique segments
DT_RFM_unique = DT_RFM.groupby('RFM_Segment_Concat')['RFM_Segment_Concat'].nunique()
print('N° of unique segments:', DT_RFM_unique.sum())

# Calculate RFM_Score
DT_RFM['RFM_Score'] = DT_RFM[['R','F','M']].sum(axis=1)
print('--------------------------')
print(DT_RFM['RFM_Score'].head())


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

