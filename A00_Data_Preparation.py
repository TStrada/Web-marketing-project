# -*- coding: utf-8 -*-
"""Data Preparation
"""

# Commented out IPython magic to ensure Python compatibility.
#Importation of the libraries
import pandas as pd
import numpy as np
import re
from scipy import stats
from datetime import timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set_theme(style="whitegrid")
import datetime

BASE_PATH = '/content/drive/MyDrive/Progetto_Web/Dataset_Script/Dataset/'

# df1: contiene informazioni sugli abbonamenti fedeltà di ciascun account cliente
df1 = pd.read_csv(BASE_PATH + 'raw_1_cli_fid.csv', sep = ';', na_values = '', encoding = 'latin-1')
# df2: contiene informazioni su ciascun account cliente e descrive le caratteristiche di esse, tra la prima tabella e la seconda ci sono dei duplicati, 
# per esempio, perché un cliente può avere più tessere fedeltà o la stessa tessera fedeltà può appartenere a più clienti
df2 = pd.read_csv(BASE_PATH + 'raw_2_cli_account.csv', sep = ';', na_values = '', encoding = 'latin-1')
# df3: contiene informazioni sull'indirizzo corrispondente a un account cliente
df3 = pd.read_csv(BASE_PATH + 'raw_3_cli_address.csv', sep = ';', na_values = '', encoding = 'latin-1')
# df4: contiene informazioni sulle politiche sulla privacy accettate da ciascun cliente
df4 = pd.read_csv(BASE_PATH + 'raw_4_cli_privacy.csv', sep = ';', na_values = '', encoding = 'latin-1')
# df7: contiene le transazioni di acquisto e rimborso di ciascun cliente, è una delle parti più cospicue di questa base di dati
df7 = pd.read_csv(BASE_PATH + 'raw_7_tic.csv', sep = ';', na_values = '', encoding = 'latin-1')

"""# df1"""

#Check for duplicates
print('Check for duplicates')
print(f'Number of rows duplicated = {len(df1[df1.duplicated()])}')
print(f"Number of ID_FID's values duplicated = {len(df1[df1.duplicated('ID_FID')])}")
print(f"Number of ID_CLI's values duplicated = {len(df1[df1.duplicated('ID_CLI')])}")
print(f"Number of duplicates for combination ID_CLI and ID_FID = {len(df1[df1.duplicated(subset = {'ID_FID', 'ID_CLI'})])}")

#Formatting dates
df1['DT_ACTIVE'] = pd.to_datetime(df1['DT_ACTIVE'])
#Formatting boolean as category
df1 = df1.astype({'TYP_CLI_FID': 'category', 'STATUS_FID': 'category'})

#CONSISTENCY CHECK on df1: number of fidelity subscriptions per client
#Count the subscriptions for each client
num_fid_x_cli = df1.groupby('ID_CLI', as_index=False)[['ID_FID', 'DT_ACTIVE']].nunique()
num_fid_x_cli.rename(columns={"ID_FID": "NUM_FIDs", "DT_ACTIVE": "NUM_DATEs"}, inplace = True)
tot_id_cli = len(num_fid_x_cli)
tot_id_cli

#Compute the distribution of number of subscriptions
dist_num_fid_x_cli = num_fid_x_cli.groupby(by=['NUM_FIDs', 'NUM_DATEs'], as_index=False)[['ID_CLI']].nunique()
dist_num_fid_x_cli.rename(columns={"ID_CLI": "TOT_CLIs"}, inplace = True)
dist_num_fid_x_cli['PERCENT_CLIs'] = dist_num_fid_x_cli['TOT_CLIs']/tot_id_cli
#NOTE: there are clients with multiple fidelity subscriptions!
dist_num_fid_x_cli

#Let examine in details clients with multiple subscriptions
num_fid_x_cli[num_fid_x_cli['NUM_FIDs'] == 3]

#Each subscription can have different dates
df1[df1['ID_CLI'] == 621814]

#Combining information
#From first subscription  --> registration date, store for registration
#From last subscription   --> type of fidelity, status
#From subscriptions count --> number of subscriptions made
df1_min_date = df1.groupby('ID_CLI', as_index = False)['DT_ACTIVE'].min()
df_1_cli_fid_first = df1_min_date.merge(df1, how = 'left')
df_1_cli_fid_first.sort_values('ID_FID', inplace = True)
duplicates_min = df_1_cli_fid_first[df_1_cli_fid_first.duplicated('ID_CLI')]
index_min = list(duplicates_min.index.values)
df_1_cli_fid_first.drop(index_min, inplace = True)
df1_max_date = df1.groupby('ID_CLI', as_index = False)['DT_ACTIVE'].max()
df_1_cli_fid_last = df1_max_date.merge(df1, how = 'left')
df_1_cli_fid_last.sort_values('ID_FID', inplace = True, ascending = False)
duplicates_max = df_1_cli_fid_last[df_1_cli_fid_last.duplicated('ID_CLI')]
index_max = list(duplicates_max.index.values)
df_1_cli_fid_last.drop(index_max, inplace = True)
df_1_cli_fid_last.rename(columns={'COD_FID' : 'LAST_COD_FID', 'TYP_CLI_FID' : 'LAST_TYP_CLI_FID', 'STATUS_FID': 'LAST_STATUS_FID', 'DT_ACTIVE' : 'LAST_DT_ACTIVE'}, inplace = True)
df_1_cli_fid_first.rename(columns = {'ID_NEG' : 'FIRST_ID_NEG', 'DT_ACTIVE' : 'FIRST_DT_ACTIVE'}, inplace = True)
df_1_cli_fid_clean = pd.merge(df_1_cli_fid_last[['ID_CLI', 'ID_FID', 'LAST_COD_FID', 'LAST_TYP_CLI_FID', 'LAST_STATUS_FID', 'LAST_DT_ACTIVE']],df_1_cli_fid_first[['ID_CLI','FIRST_ID_NEG', 'FIRST_DT_ACTIVE']], on='ID_CLI', how='left')
df_1_cli_fid_clean = df_1_cli_fid_clean.merge(num_fid_x_cli[['ID_CLI', 'NUM_FIDs']], on='ID_CLI', how='left')
df_1_cli_fid_clean = df_1_cli_fid_clean.astype({'NUM_FIDs': 'category'})

df_1_cli_fid_clean.head()

"""## Explore columns"""

#Variable LAST_COD_FID
#Compute distribution
df1_dist_codfid = df_1_cli_fid_clean.groupby('LAST_COD_FID', as_index = False)[['ID_CLI']].nunique()
df1_dist_codfid.rename(columns = {'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df1_dist_codfid['PERCENT'] = df1_dist_codfid['TOT_CLIs']/df1_dist_codfid['TOT_CLIs'].sum()
df1_dist_codfid.sort_values('PERCENT', inplace = True, ascending = False)

#Plot distribution
plt.bar(df1_dist_codfid['LAST_COD_FID'], df1_dist_codfid['TOT_CLIs'],align='center') # A bar chart
plt.xlabel('LAST_COD_FID')
plt.ylabel('TOT_CLIs')
plt.grid(linestyle='-', linewidth=0.2)
plt.show()


#Variable LAST_TYP_CLI_FID
#Compute distribution
df1_dist_clifid = df_1_cli_fid_clean.groupby('LAST_TYP_CLI_FID', as_index = False)[['ID_CLI']].nunique()
df1_dist_clifid.rename(columns = {'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df1_dist_clifid['PERCENT'] = df1_dist_clifid['TOT_CLIs']/df1_dist_clifid['TOT_CLIs'].sum()
df1_dist_clifid.sort_values('PERCENT', inplace = True, ascending = False)

#Plot distribution
plt.bar(df1_dist_clifid['LAST_TYP_CLI_FID'], df1_dist_clifid['TOT_CLIs'],align='center') # A bar chart
plt.xlabel('LAST_TYP_CLI_FID')
plt.ylabel('TOT_CLIs')
plt.grid(linestyle='-', linewidth=0.2)
plt.show()

#Variable LAST_STATUS_FID
#Compute distribution
df1_dist_statusfid = df_1_cli_fid_clean.groupby('LAST_STATUS_FID', as_index = False)[['ID_CLI']].nunique()
df1_dist_statusfid.rename(columns = {'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df1_dist_statusfid['PERCENT'] = df1_dist_statusfid['TOT_CLIs']/df1_dist_statusfid['TOT_CLIs'].sum()
df1_dist_statusfid.sort_values('PERCENT', inplace = True, ascending = False)

#Plot distribution
plt.bar(df1_dist_statusfid['LAST_STATUS_FID'], df1_dist_statusfid['TOT_CLIs'],align='center') # A bar chart
plt.xlabel('LAST_STATUS_FID')
plt.ylabel('TOT_CLIs')
plt.grid(linestyle='-', linewidth=0.2)
plt.show()

#Variable NUM_FIDs
#Compute distribution
df1_dist_numfid = df_1_cli_fid_clean.groupby('NUM_FIDs', as_index = False)[['ID_CLI']].nunique()
df1_dist_numfid.rename(columns = {'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df1_dist_numfid['PERCENT'] = df1_dist_numfid['TOT_CLIs']/df1_dist_numfid['TOT_CLIs'].sum()
df1_dist_numfid.sort_values('PERCENT', inplace = True, ascending = False)

#Plot distribution
plt.bar(df1_dist_numfid['NUM_FIDs'], df1_dist_numfid['TOT_CLIs'],align='center') # A bar chart
plt.xlabel('NUM_FIDs')
plt.ylabel('TOT_CLIs')
plt.grid(linestyle='-', linewidth=0.2)
plt.show()

#FINAL REVIEW df_1_clean
df_1_cli_fid_clean.info()

df_1_cli_fid_clean.describe()

df1 = df_1_cli_fid_clean
df1.to_csv(BASE_PATH + 'df1.csv', index=False)
df1.to_csv(BASE_PATH + 'df1_zip.csv', index=False, compression='gzip')

#Check for duplicates
print('Check for duplicates')
print(f'Number of rows duplicated = {len(df2[df2.duplicated()])}')
print(f"Number of ID_CLI's values duplicated = {len(df2[df2.duplicated('ID_CLI')])}")

# format numerical categories as categorical 
df2 = df2.astype({'TYP_CLI_ACCOUNT': 'category'})

# CLEANING MISSING VALUES in df_2
# filling missing value using fillna() 
df2["W_PHONE"] = df2["W_PHONE"].fillna(0)
print(sum(df2["W_PHONE"].isnull()))
df2["EMAIL_PROVIDER"] = df2["EMAIL_PROVIDER"].fillna("(missing)")
print(sum(df2["EMAIL_PROVIDER"].isnull()))
df2["TYP_JOB"] = df2["TYP_JOB"].fillna("(missing)")
print(sum(df2["TYP_JOB"].isnull()))

# tutti gli id in df1 sono anche in df2
df1['ID_CLI'].isin(df2['ID_CLI']).value_counts()
# True= comune
# False= diverso

df2['EMAIL_PROVIDER']

# ci sono molti valori diversi per EMAIL_PROVIDER
pd.set_option("display.max_rows", None)
df2['EMAIL_PROVIDER'].value_counts()
df2['EMAIL_PROVIDER'].value_counts(normalize=True) * 100 # percentuale

df2['EMAIL_PROVIDER'].value_counts().head(20)

# calcolo probabilità cumulate
from collections import Counter
items = Counter(df2['ID_CLI']).keys()
out = np.cumsum(df2['EMAIL_PROVIDER'].value_counts()) / len(items)
out

name_providers = []
for idx, name in enumerate(df2['EMAIL_PROVIDER'].value_counts().head(20).index.tolist()):
    name_providers.append(name)
name_providers

# vengono mantenuti i primi 20 che hanno una frequenza cumulata < 85% e gli altri vengono nominati come "other"
for i in df2['EMAIL_PROVIDER']:
  if (i not in name_providers):
    df2['EMAIL_PROVIDER'] = df2['EMAIL_PROVIDER'].replace(i,'other')

df2['EMAIL_PROVIDER'].value_counts()

df2.to_csv(BASE_PATH + 'df2.csv', index=False)
df2.to_csv(BASE_PATH + 'df2_zip.csv', index=False, compression='gzip')

#Check for duplicates
print('Check for duplicates')
print(f'Number of rows duplicated = {len(df3[df3.duplicated()])}')
print(f"Number of ID_ADDRESS's values duplicated = {len(df3[df3.duplicated('ID_ADDRESS')])}") 
# ci sono dei duplicati

df3 = df3.drop_duplicates()

#Check for duplicates
print('Check for duplicates')
print(f'Number of rows duplicated = {len(df3[df3.duplicated()])}')
print(f"Number of ID_ADDRESS's values duplicated = {len(df3[df3.duplicated('ID_ADDRESS')])}")

df3.dtypes

# format numerical categories as categorical 
df3 = df3.astype({'CAP': 'category'})

# CLEANING MISSING VALUES in df_3
df3.isnull().sum()

df3.dropna(axis='index', inplace=True)

df3.head()

# ci sono degli address che non sono mappati in df2
df2['ID_ADDRESS'].isin(df3['ID_ADDRESS']).value_counts()
#df3['ID_ADDRESS'].isin(df2['ID_ADDRESS']).value_counts()

df3.to_csv(BASE_PATH + 'df3.csv', index=False)
df3.to_csv(BASE_PATH + 'df3_zip.csv', index=False, compression='gzip')

"""# df4"""

#Check for duplicates
print('Check for duplicates')
print(f'Number of rows duplicated = {len(df4[df4.duplicated()])}')
print(f"Number of ID_CLI's values duplicated = {len(df4[df4.duplicated('ID_CLI')])}") 
# NON ci sono duplicati

df4.head()

# format boolean as categorical 
df4 = df4.astype({'FLAG_PRIVACY_1': 'category'})
df4 = df4.astype({'FLAG_PRIVACY_2': 'category'})
df4 = df4.astype({'FLAG_DIRECT_MKT': 'category'})

df1['ID_CLI'].isin(df4['ID_CLI']).value_counts()
df4['ID_CLI'].isin(df1['ID_CLI']).value_counts()
# tutti gli ID_CLI in df1 sono anche in df4 e viceversa

df4.to_csv(BASE_PATH + 'df4.csv', index=False)
df4.to_csv(BASE_PATH + 'df4__zip.csv', index=False, compression='gzip')

"""# df7"""

df7.head()

df7.info()

df7.describe()

df7.shape

#Formatting dates and times
df7['DATETIME'] = pd.to_datetime(df7['DATETIME'])
df7['ORA'] = df7['DATETIME'].dt.hour
df7['DATA'] = df7['DATETIME'].dt.date
df7['DATA'] = pd.to_datetime(df7['DATA'])

#Formatting boolean as category
df7 = df7.astype({'DIREZIONE': 'category'})
#Formatting numerical categories as category
df7 = df7.astype({'COD_REPARTO': 'category'})

df7['IMPORTO_LORDO'] = df7['IMPORTO_LORDO'].replace(',','.', regex=True)
df7 = df7.astype({'IMPORTO_LORDO': 'float'})

df7['SCONTO'] = df7['SCONTO'].replace(',','.', regex=True)
df7 = df7.astype({'SCONTO': 'float'})

#Elimino le righe che hanno DIREZIONE = 1 e con IMPORTO_LORDO negativo perché non sono casi possibili, sono degli errori.
df7.drop(df7.index[(df7['DIREZIONE'] == 1) & (df7['IMPORTO_LORDO'] < 0)], inplace = True)

#Creo la colonna che considera l'importo netto perché così ne elimino due (le eliminiamo alla fine per via di altre operazioni)
df7['IMPORTO_NETTO'] = df7['IMPORTO_LORDO'] - df7['SCONTO']
df7.reset_index(inplace = True, drop = True)

df7.drop(df7.index[(df7['IMPORTO_NETTO'] < 0)], inplace = True)
df7.reset_index(inplace = True, drop = True)

len(df7[(df7['IMPORTO_NETTO'] < 0)])

#Controllo se ci sono delle righe duplicate in base a tutte le colonne tranne l'IMPORTO_NETTO
subset= df7[['ID_SCONTRINO', 'ID_CLI', 'ID_NEG', 'ID_ARTICOLO', 'DIREZIONE', 'COD_REPARTO', 'DATETIME', 'ORA', 'DATA']]
print('Check for duplicates:')
print(f'Number of duplicated purchases = {len(subset[subset.duplicated()])}')
#I duplicati sono normali perché se compro più quantità di uno stesso prodotto nello stesso scontrino
#viene duplicata la riga relativa perché non tiene conto delle quantità

len(df7[(df7['IMPORTO_NETTO'] == 0)])

# si nota che 12192 righe hanno importo netto pari a 0 poichè è stato applicato uno sconto del 100%, ovvero l'importo lordo equivale allo 
# sconto applicato.
df7[(df7['IMPORTO_NETTO'] == 0)]

#In seguito controllo i duplicati per vedere se c'è qualcosa che non torna
df7[(df7['ID_SCONTRINO'] == '51183329701/05/181123338499039') & (df7['ID_CLI'] == 310038) & (df7['ID_ARTICOLO'] == 35573020)]
#In questo caso, e non è l'unico, lo stesso articolo viene acquistato dallo stesso cliente nello 
#stesso scontrino dello stesso giorno ma l'IMPORTO_NETTO è diverso

#A meno di particoli offerte, di cui non siamo informati, non puoi avere lo stesso prodotto nello
#stesso scontrino venduto con valori di IMPORTO_NETTO diversi.
#Salvo a parte le righe che sono identiche quindi quelle che indicano un acquisto di più prodotti
#uguali in una stessa volta.
identical = df7[df7.duplicated(keep = False)]

#Elimino da tutto il dataset i duplicati identici e non ne tengo nessuno
df7.drop_duplicates(inplace = True, keep = False)
df7.reset_index(inplace = True, drop = True)

#Le righe rimaste non possono avere duplicati identici, quindi sono quelle con problemi, di queste non
#ne tengo nessuna perché non so qual è il prezzo esatto
df7.drop_duplicates(subset = ['ID_SCONTRINO', 'ID_CLI', 'ID_NEG', 'ID_ARTICOLO', 'COD_REPARTO', 'DIREZIONE', 'DATETIME', 'ORA', 'DATA'], inplace = True, keep = False)
df7.reset_index(inplace = True, drop = True)

#Riunisco al dataset quelli identici perché li volevo tenere
df7 = df7.append(identical, ignore_index=True)

"""## Overview"""

#Compute aggregate
df7_overview = pd.DataFrame({'MIN_DATE' : [df7['DATA'].min()], 'MAX_DATE' : [df7['DATA'].max()], 
                             'TOT_TICs' : [df7['ID_SCONTRINO'].nunique()], 'TOT_CLIs' : [df7['ID_CLI'].nunique()]})
df7_overview
#Il periodo di tempo va dal 01-05-2018 al 30-04-2019

#Variable DIREZIONE
#Compute aggregate
df7_dist_direction = df7.groupby('DIREZIONE', as_index = False)[['ID_SCONTRINO', 'ID_CLI']].nunique()
df7_dist_direction.rename(columns = {'ID_SCONTRINO' : 'TOT_TICs', 'ID_CLI' : 'TOT_CLIs'}, inplace = True)
#Sono i risultati delle divisioni, li ho messi così perché mi venivano NaN
df7_dist_direction['PERCENT_TICs'] = [0.09036657031066045, 0.9096334296893396]
df7_dist_direction['PERCENT_CLIs'] = [0.2197865399483321, 1.0]
df7_dist_direction

# Variable ORA
#Compute aggregate
df7_dist_hour = df7.groupby(['ORA', 'DIREZIONE'])[['ID_SCONTRINO', 'ID_CLI']].nunique()
df7_dist_hour.reset_index(inplace = True)
df7_dist_hour.rename(columns = {'ID_SCONTRINO' : 'TOT_TICs', 'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df7_dist_direction.rename(columns = {'TOT_TICs' : 'ALL_TOT_TICs', 'TOT_CLIs' : 'ALL_TOT_CLIs'}, inplace = True)
df7_dist_hour = pd.merge(df7_dist_hour, df7_dist_direction[['DIREZIONE', 'ALL_TOT_TICs', 'ALL_TOT_CLIs']], on = 'DIREZIONE', how = 'left')
df7_dist_hour['PERCENT_TICs'] = df7_dist_hour.TOT_TICs/df7_dist_hour.ALL_TOT_TICs
df7_dist_hour['PERCENT_CLIs'] = df7_dist_hour.TOT_CLIs/df7_dist_hour.ALL_TOT_CLIs
df7_dist_hour.drop(columns = {'ALL_TOT_TICs', 'ALL_TOT_CLIs'}, inplace = True)
df7_dist_hour.drop([0, 2, 4, 6], inplace =True)
df7_dist_hour.reset_index(inplace = True, drop = True)
df7_dist_hour
#plot

#Variable COD_REPARTO
#Compute aggregate
df7_dist_dep = df7.groupby(['COD_REPARTO', 'DIREZIONE'])[['ID_SCONTRINO', 'ID_CLI']].nunique()
df7_dist_dep.rename(columns = {'ID_SCONTRINO' : 'TOT_TICs', 'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df7_dist_dep.reset_index(inplace = True)
df7_dist_dep = pd.merge(df7_dist_dep, df7_dist_direction[['DIREZIONE', 'ALL_TOT_TICs', 'ALL_TOT_CLIs']], how = 'left', on = 'DIREZIONE')
df7_dist_dep['PERCENT_TICs'] = df7_dist_dep.TOT_TICs/df7_dist_dep.ALL_TOT_TICs
df7_dist_dep['PERCENT_CLIs'] = df7_dist_dep.TOT_CLIs/df7_dist_dep.ALL_TOT_CLIs
df7_dist_dep.drop(columns = {'ALL_TOT_TICs', 'ALL_TOT_CLIs'}, inplace = True)
df7_dist_dep
#plot

#Variable ID_ARTICOLO
#Compute aggregate
df7_dist_articolo = df7.groupby(['ID_ARTICOLO', 'DIREZIONE'])[['ID_SCONTRINO', 'ID_CLI']].nunique()
df7_dist_articolo.rename(columns = {'ID_SCONTRINO' : 'TOT_TICs', 'ID_CLI' : 'TOT_CLIs'}, inplace = True)
df7_dist_articolo.reset_index(inplace = True)
df7_dist_articolo.head()
#plot

#Distribution of customers by number of purchases
#Elimino quelli con DIREZIONE = -1, perché mi interessa che abbiano acquistato, se anche restituiscono ma fanno più acquisti sono clienti.
df7_dist_customer = df7.groupby(['ID_CLI', 'DIREZIONE'])[['ID_SCONTRINO']].nunique()
df7_dist_customer.reset_index(inplace = True)
df7_dist_customer.drop(df7_dist_customer.index[df7_dist_customer['ID_SCONTRINO'] == 0], inplace = True)
df7_dist_customer.drop(df7_dist_customer.index[df7_dist_customer['DIREZIONE'] == -1], inplace = True)
df7_dist_customer.rename(columns = {'ID_SCONTRINO' : 'TOT_SCONTRINI'}, inplace = True)
df7_dist_customer.reset_index(inplace = True, drop = True)
#df7_dist_customer = df7_dist_customer.groupby('TOT_SCONTRINI', as_index = False)[['ID_CLI']].count()
#df7_dist_customer.rename(columns = {'ID_CLI' : 'TOT_CLI'}, inplace = True)
df7_dist_customer.head()

df7_dist_customer.sort_values(['TOT_SCONTRINI'], ascending = [False])
#I 3 id_cliente che registrano il numero maggiore di acquisti sono: 376925, 117212, 248975 con rispettivamente 177, 155, 154 acquisti

df7_plot_dist_customer = pd.DataFrame({'Numero di acquisti':['1 o più', '2 o più', '3 o più', '4 o più', '5 o più', '6 o più'], 
                                       'Numero di clienti': [len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 1]), 
                                                             len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 2]), 
                                                             len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 3]), 
                                                             len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 4]), 
                                                             len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 5]), 
                                                             len(df7_dist_customer[df7_dist_customer['TOT_SCONTRINI'] >= 6])]})
ax = df7_plot_dist_customer.plot.bar(x='Numero di acquisti', y='Numero di clienti', rot=0)

"""### The days for next purchase curve"""

#Considero che ne fa uno al giorno
df7_day_next_purchase = df7.groupby('ID_CLI', as_index = False).agg(PRIMO_ACQUISTO = ('DATA', 'min'), 
                                                                    ULTIMO_ACQUISTO = ('DATA', 'max'), NUMERO_ACQUISTI = ('DATA', 'nunique'))
df7_day_next_purchase

df_7_next = df7.sort_values(by = 'DATA')
df_7_next.reset_index(inplace = True, drop = True)
df7_day_next = df_7_next.groupby(['ID_CLI', 'DATA']).agg(PRIMO_ACQUISTO = ('DATA', 'min'))
df7_day_next.reset_index(inplace = True)
df7_day_next.drop(['PRIMO_ACQUISTO'], axis = 1, inplace = True)
df7_day_next = df7_day_next.groupby('ID_CLI')[['DATA']].nth(-2)
df7_day_next.reset_index(inplace = True)
df7_day_next.rename(columns = {'DATA' : 'PENULTIMO_ACQUISTO'}, inplace = True)
df7_day_next

df7_day_next_purchase = pd.merge(df7_day_next_purchase, df7_day_next, on = 'ID_CLI', how = 'left')
df7_day_next_purchase['NEXT_PURCHASE_DAY'] = (df7_day_next_purchase['ULTIMO_ACQUISTO'] - df7_day_next_purchase['PENULTIMO_ACQUISTO']).dt.days
df7_day_next_purchase['NEXT_PURCHASE_DAY'] = df7_day_next_purchase['NEXT_PURCHASE_DAY'].fillna(0)
df7_day_next_purchase

ax = sns.kdeplot(data = df7_day_next_purchase, x = 'NEXT_PURCHASE_DAY', cumulative = True)
plt.xticks([0,44,69,93,150,200,300,350])
plt.vlines(x = 69, ymin = 0, ymax = 0.85, color='r', linestyle='--')
plt.vlines(x = 49, ymin = 0, ymax = 0.80, color='y', linestyle='--')
plt.vlines(x = 99, ymin = 0, ymax = 0.90, color='y', linestyle='--')
plt.ylabel('%Customers')
plt.savefig('NPDC.png')

print('Last purchase: ' + str(df7.DATETIME.max()))
print('First purchase: ' + str(df7.DATETIME.min()))

## Set last date as point cut to calculate the day next purchase 
max_date = df7_day_next_purchase.NEXT_PURCHASE_DAY.max()

print('Summary: ', df7_day_next_purchase.NEXT_PURCHASE_DAY.describe())
print('80% ', np.percentile(a = df7_day_next_purchase.NEXT_PURCHASE_DAY, q = 80))
print('85% ', np.percentile(a = df7_day_next_purchase.NEXT_PURCHASE_DAY, q = 85))
print('90% ', np.percentile(a = df7_day_next_purchase.NEXT_PURCHASE_DAY, q = 90))

"""### df7_churn"""

# NB !!!
# Data la "the days for next purchase curve" si nota per circa il 90% dei clienti passano circa 99 giorni 
# per il successivo acquisto, per l'80% passano 49 giorni
# Questo significa che dato che la maggior parte delle persone acquista dopo 69 giorni, 
# se un cliente non dovesse acquistare questo deve essere un campanello d'allarme, 
# il cliente potrebbe essere un possibile churn

# creiamo la colonna CHURN supponendo: 0 = NON CHURN, 1 = CHURN
# si suppone un periodo di holdout di circa 69 giorni, pertanto tutti gli acquisti effettuati 

# quindi consideriamo un cliente come churn i clienti che non riacquistano entro due mesi e mezzo (entro il 2019/02/19)

#creiamo la colonna dei churner
df_churn1 = df7.groupby(['ID_CLI'])[['DATA']].max()
df_churn1.rename(columns = {'DATA' : 'LAST_PURCHASE_DATE'}, inplace = True)
df_churn1.reset_index(inplace = True)

df_churn2 = df7.groupby(['ID_CLI'])[['IMPORTO_LORDO']].sum()
df_churn2.rename(columns = {'IMPORTO_LORDO' : 'TOTAL_PURCHASE'}, inplace = True)
df_churn2.reset_index(inplace = True)

df_churn3 = df7.groupby(['ID_CLI', 'DIREZIONE'])[['ID_SCONTRINO']].nunique()
df_churn3.reset_index(inplace = True)
df_churn3.drop(df_churn3.index[df_churn3['ID_SCONTRINO'] == 0], inplace = True)
df_churn3.drop(df_churn3.index[df_churn3['DIREZIONE'] == -1], inplace = True)
df_churn3.rename(columns = {'ID_SCONTRINO' : 'NUMBER_OF_PURCHASE'}, inplace = True)
df_churn3.reset_index(inplace = True, drop = True)

df_churn = pd.merge(left=df_churn1, right=df_churn2, how='left', on='ID_CLI')
pd.merge(left=df_churn, right=df_churn3, how='left', on='ID_CLI')

from datetime import date

CHURN =[]
# 1 = TRUE, 0 = FALSE
for i in range(len(df_churn['LAST_PURCHASE_DATE'])):
  if (df_churn['LAST_PURCHASE_DATE'][i] < date(2019, 2, 19)):
    CHURN.append(1)
  else:
    CHURN.append(0)

df_churn['CHURN'] = CHURN

df_churn

"""### Final review"""

df7.drop(columns = ['IMPORTO_LORDO', 'SCONTO'], inplace = True) # siccome abbiamo aggiunto la colonna IMPORTO_NETTO, queste due colonne non ci servono più

df7.info()

df7.describe()

"""### Save"""

df7.to_csv(BASE_PATH + 'df7.csv', index=False)
df7.to_csv(BASE_PATH + 'df7_zip.csv', index=False, compression='gzip')

df7_churn = pd.merge(left=df_churn, right=df7, how='left', on='ID_CLI')
df7_churn.to_csv(BASE_PATH + 'df7_churn.csv', index=False)
df7_churn.to_csv(BASE_PATH + 'df7_churn_zip.csv', index=False, compression='gzip')
