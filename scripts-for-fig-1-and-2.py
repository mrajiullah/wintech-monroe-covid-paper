#!/usr/bin/env python
# coding: utf-8

# ### Load libraries 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from matplotlib.pyplot import figure
import os
from matplotlib.ticker import MaxNLocator


# ### Load datasets

# In[10]:


dataset_loc="wintech-webdatasets.csv"
df= pd.read_csv(dataset_loc,index_col=[0])
df['date'] = pd.to_datetime(df['Timestamp'],unit='s')
df["month"]=df['date'].dt.month
df["week"]=df['date'].dt.week
df.pageLoadTime=df.pageLoadTime/1000


# In[ ]:






# ### Figure-1

# In[23]:


df_feb=df[(df.Ops!='op2 (IT)') & (df.month==2)]
feb_mean=df_feb.groupby(['url','Ops'])['pageLoadTime'].mean()
feb_mean.reset_index().to_csv('feb1.csv', columns=['url','Ops', 'pageLoadTime'], index=False)
df_feb_mean = pd.read_csv("feb1.csv")

df_mar=df[df.month==3]
mar_mean=df_mar.groupby(['url','Ops'])['pageLoadTime'].mean()
mar_mean.reset_index().to_csv('mar1.csv', columns=['url','Ops', 'pageLoadTime'], index=False)
df_mar_mean = pd.read_csv("mar1.csv")

df_mar_mean['increase']=((df_mar_mean.pageLoadTime-df_feb_mean.pageLoadTime)/df_mar_mean.pageLoadTime)*100
del df_mar_mean["pageLoadTime"]
x__ = df_mar_mean.pivot("url", "Ops", "increase")

order=["op1 (SE)", "op2 (SE)", "op3 (SE)","op1 (NO)",  "op2 (NO)","op3 (NO)", "op1 (IT)",  "op1 (ES)","op2 (ES)"]
x = x__[["op1 (SE)", "op2 (SE)", "op3 (SE)","op1 (NO)",  "op2 (NO)","op3 (NO)", "op1 (IT)",  "op1 (ES)","op2 (ES)"]]

plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.labelsize']=34
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['legend.fontsize']=28
plt.rc('text', usetex=True)
plt.rcParams.update({'font.size': 24})

f, ax = plt.subplots(figsize=(15, 12))
sns.heatmap(x, annot=True, fmt=".1f", linewidths=.5, ax=ax)
degrees = 45
plt.xticks(rotation=degrees)

degrees = 45
plt.yticks(rotation=degrees)
plt.xlabel("Operators")
plt.tight_layout()
#plt.savefig("xx.pdf")
plt.show()


# ### Figure-2

# In[24]:


plt.rcParams['figure.dpi'] = 100
# plt.rcParams['axes.labelsize']=34
plt.rcParams['axes.labelsize']=34
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['legend.fontsize']=28


# In[25]:


df[(df.Ops=="op1 (IT)") ].groupby(["week","url"]).pageLoadTime.mean().reset_index().to_csv(
            'temp.csv', columns=['week','url', 'pageLoadTime'], index=False)

df_=pd.read_csv("temp.csv")
df_1 = df_.groupby(["url"])["pageLoadTime"].apply(
    list).reset_index(name='new')

df_1.set_index("url", inplace=True)

df_2 = df_1.transpose()
df_2.reset_index()

z = []
weeks = iter([5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22])

for k, row in df_2.iterrows():
    for j1, j2, j3, j4, j5, j6, j7, j8, j9, j10 in zip(list(np.array(row.claro).flat), list(np.array(row.facebook).flat), list(np.array(row.fantia).flat), list(np.array(row["free-power-point-templates"]).flat), list(np.array(row.google).flat), list(np.array(row.keycdn).flat), list(np.array(row.litespeedtech).flat), list(np.array(row.net).flat), list(np.array(row.youtube).flat), list(np.array(row.zaobao).flat)):
        z.append({'week': next(weeks), 'claro': j1, 'facebook': j2, 'fantia': j3, 'free-power-point-templates': j4,
                  'google': j5, 'keycdn': j6, 'litespeedtech': j7, 'net': j8, 'youtube': j9, 'zaobao': j10})

result = pd.DataFrame(z)

result.set_index("week", inplace=True)

t=result.transpose()
t['Total'] = t.sum(axis=1)

t1=t.sort_values(by=['Total'], ascending=True).transpose()

t1.drop(t1.tail(1).index,inplace=True)

fig = plt.figure(figsize=(9, 7))
ax = plt.subplot(111)

t1.plot(kind="area",ax=ax)

plt.xlabel("Date")
plt.ylabel("PLT [s]")

ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False, prop={"size":16})
ax.xaxis.set_major_locator(MaxNLocator(integer=True))



labels = [item.get_text() for item in ax.get_xticklabels()]


labels[1] = '2020-02-03'
labels[2] = '2020-02-17'
labels[3] = '2020-03-02'
labels[4] = '2020-03-17'
labels[5] = '2020-03-30'
labels[6] = '2020-04-20'
labels[7] = '2020-05-01'
labels[8] = '2020-05-11'
labels[9] = '2020-05-25'
#labels[9] = '2020-01-27'
ax.set_xticklabels(labels)
degrees = 45
plt.xticks(rotation=degrees)
plt.tight_layout()
plt.show()
#plt.savefig("xy.pdf")


# In[26]:


df[(df.Ops=="op1 (SE)") ].groupby(["week","url"]).pageLoadTime.mean().reset_index().to_csv(
            'temp.csv', columns=['week','url', 'pageLoadTime'], index=False)

df_=pd.read_csv("temp.csv")
df_1 = df_.groupby(["url"])["pageLoadTime"].apply(
    list).reset_index(name='new')

df_1.set_index("url", inplace=True)

df_2 = df_1.transpose()
df_2.reset_index()

z = []
weeks = iter([5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22])

for k, row in df_2.iterrows():
    for j1, j2, j3, j4, j5, j6, j7, j8, j9, j10 in zip(list(np.array(row.claro).flat), list(np.array(row.facebook).flat), list(np.array(row.fantia).flat), list(np.array(row["free-power-point-templates"]).flat), list(np.array(row.google).flat), list(np.array(row.keycdn).flat), list(np.array(row.litespeedtech).flat), list(np.array(row.net).flat), list(np.array(row.youtube).flat), list(np.array(row.zaobao).flat)):
        z.append({'week': next(weeks), 'claro': j1, 'facebook': j2, 'fantia': j3, 'free-power-point-templates': j4,
                  'google': j5, 'keycdn': j6, 'litespeedtech': j7, 'net': j8, 'youtube': j9, 'zaobao': j10})



result = pd.DataFrame(z)

result.set_index("week", inplace=True)

t=result.transpose()
t['Total'] = t.sum(axis=1)

t1=t.sort_values(by=['Total'], ascending=True).transpose()

t1.drop(t1.tail(1).index,inplace=True)

fig = plt.figure(figsize=(9, 7))
ax = plt.subplot(111)

t1.plot(kind="area",ax=ax)
#plt.title("TIM (IT)")
plt.xlabel("Date")
plt.ylabel("PLT [s]")

ax.legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=2,
            borderaxespad=0, frameon=False, prop={"size":16})
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#ax.set_yscale('log')


labels = [item.get_text() for item in ax.get_xticklabels()]


labels[1] = '2020-02-03'
labels[2] = '2020-02-17'
labels[3] = '2020-03-02'
labels[4] = '2020-03-17'
labels[5] = '2020-03-30'
labels[6] = '2020-05-01'
labels[7] = '2020-05-11'
labels[8] = '2020-05-25'
labels[9] = '2020-05-25'

ax.set_xticklabels(labels)
degrees = 45
plt.xticks(rotation=degrees)
plt.tight_layout()
#plt.savefig("xz.pdf")
plt.show()






