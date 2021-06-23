#!/usr/bin/env python
# coding: utf-8

# # Data Analysis and Visualization: Netflix Data

# This dataset consists of tv shows and movies available on Netflix as of 2019. 
# 
# In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.
# 
# Exploring the data for better insights
# 
# 

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from termcolor import colored

import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('netflix_titles.csv')
df.head()


# In[4]:


print(colored('There are {} rows and {} columns in the dataset.'.
             format(df.shape[0],df.shape[1]),attrs=['bold']))


# In[5]:


plt.figure(figsize=(10,5))
sns.heatmap(df.isnull())

for i in df.columns:
    null_rate=df[i].isna().sum()/len(df)*100
    if null_rate>0:
        print("{}'s null rate :{}%".format(i,round(null_rate,2)))


# **The columns 'director', 'cast' , 'country', 'date_added' and 'rating' have missing values.**

# In the above data:
# 
# I choose to drop the **'director' and 'cast'** columns completely as they have high volume of missing values and dropping these columns will not effect my visualization.

# In[6]:


df.drop(['director'],axis=1,inplace=True)
df.head()


# #### Filling all the missing values in the 'country' column with United States as Netflix was created in the USA and every show is aired on Netflix US.

# In[7]:


df['country'].replace(np.nan,'United States',inplace=True)
df['cast'].replace(np.nan,'No Data',inplace=True)

print(colored("I will drop the missing rows from the columns 'date_added' and 'rating' since these have only {} missing rows in total."
             .format(df.isnull().sum().sum()),attrs=['bold']))


# In[8]:


df.dropna(inplace=True)
print(colored("There are {} rows and {} columns after handling the missing records in the dataset.".format(
    df.shape[0],df.shape[1]),attrs=['bold']))


# In[9]:


#checking data type of the column
df.info()


# #### The Dtype of the column 'date_added' is object, converting it into datetime format.

# In[10]:


df['date_added']=pd.to_datetime(df['date_added'])
df['month_added']=df['date_added'].dt.month
df['month_name_added']=df['date_added'].dt.month_name()
df['year_added']=df["date_added"].dt.year

# Droping the column 'date_added' as it we have seperate columns 
#for 'year_added' and 'month_added'

df.drop('date_added',axis=1,inplace=True)


# In[11]:


df.head()


# ## Content type on Netflix:

# In[12]:


plt.figure(figsize=(10,5))
plt.pie(df['type'].value_counts().sort_values(),
        labels=df['type'].value_counts().index,explode=[0.1,0],
        autopct='%1.2f%%',colors=['Blue','Yellow'],shadow=True)
plt.show()


# Nearly 2/3rd of the content on netflix are movies and remaining 1/3rd of them are TV Show

# ## Top-20 countries producing most contents:

# Since there are contents that are produced in different countries sp we have to consider those too. So we have to split those rows and get the indivisual country.

# In[13]:


from collections import Counter
country_data=df['country']
country_count=pd.Series(dict(Counter(','.join(country_data).
                                     replace(' ,',',').replace(', ',',')
                                    .split(',')))).sort_values(ascending=False)


# In[14]:


top20country = country_count.head(20)
top20country


# In[57]:


from matplotlib import gridspec

fig=plt.figure(figsize=(20,7))
gs=gridspec.GridSpec(nrows=1,ncols=2,height_ratios=[6],width_ratios=[10,5])

ax=plt.subplot(gs[0])
sns.barplot(top20country.index, top20country, ax=ax,palette='gist_earth')
ax.set_xticklabels(top20country.index, rotation='90',fontweight='bold')
ax.set_title('Top 20 countries with most contents'
            , fontsize=15, fontweight='bold')

ax2=plt.subplot(gs[1])
ax2.pie(top20country, labels=top20country.index,shadow=True,
        startangle=0,colors=sns.color_palette('gist_earth',n_colors=20),
         autopct='%1.2f%%')
ax2.axis('equal')

plt.show()


# We can see that US, India, United Kingdom, Canada and France contribute 75% of the top20 countries.

# In[16]:


df_tv=df[df['type']=='TV Show']
df_movies=df[df['type']=='Movie']


# In[17]:


df.head()


# In[18]:


df_content=df['year_added'].value_counts().reset_index().rename(columns = {
    'year_added' : 'count', 'index' : 'year_added'}).sort_values('year_added')
df_content['percent']=df_content['count'].apply(lambda x:100*x/sum(df_content['count']))

df_tv1=df_tv['year_added'].value_counts().reset_index().rename(columns={
    'year_added':'count','index':'year_added'}).sort_values('year_added')
df_tv1['percent']=df_tv1['count'].apply(lambda x:100/sum(df_tv1['count']))


df_movies1=df_movies['year_added'].value_counts().reset_index().rename(
    columns={'year_added':'count','index':'year_added'}).sort_values('year_added')
df_movies1['percent']=df_movies1['count'].apply(lambda x: 100*x/sum(df_movies1['count']))

t1=go.Scatter(x=df_movies1['year_added'],y=df_movies1['count'],name='Movies',
             marker=dict(color='#a678de'))
t2=go.Scatter(x=df_tv1['year_added'],y=df_tv1['count'],name='TV Shows',
             marker=dict(color='#6ad49b'))
t3=go.Scatter(x=df_content['year_added'],y=df_content['count'],name='Total Contents',
            marker=dict(color='brown'))

data=[t1,t2,t3]

layout=go.Layout(title='Content added over the years',
                legend=dict(x=0.1,y=1.1,orientation='h'))
fig=go.Figure(data, layout=layout)
fig.show()


# In[19]:


df_content


# The growth in number of movies on netflix is much higher than that of TV shows
# 
# About 1200 new movies were added in both 2018 and 2019
# 
# The growth in content started from 2013

# ## Contents over the months:

# In[20]:


df_content=df.groupby(['month_added','month_name_added']).month_added.agg([len]).reset_index().rename(columns={'len':'count'}).sort_values('month_added').drop('month_added',axis=1)
df_content['percent']=df_content['count'].apply(lambda x:100*x/sum(df_content['count']))

df_tv2=df_tv.groupby(['month_added','month_name_added']).month_added.agg([len]).reset_index().rename(columns={'len':'count'}).sort_values('month_added').drop('month_added',axis=1)
df_tv2['percent']=df_tv2['count'].apply(lambda x:100*x/sum(df_tv2['count']))

df_movies2=df_movies.groupby(['month_added','month_name_added']).month_added.agg([len]).reset_index().rename(columns={'len':'count'}).sort_values('month_added').drop('month_added',axis=1)
df_movies2['percent']=df_movies2['count'].apply(lambda x:100*x/sum(df_movies2['count']))

t1=go.Scatter(x=df_movies2['month_name_added'],y=df_movies2['count'],name='Movies',marker=dict(color='#a678de'))
t2=go.Scatter(x=df_tv2['month_name_added'],y=df_tv2['count'],name='TV Shows',marker=dict(color='#6ad49b'))
t3=go.Scatter(x=df_content['month_name_added'],y=df_content['count'],name='Total content',marker=dict(color='Brown'))

data=[t1,t2,t3]

layout=go.Layout(title='Content added over the years',legend=dict(x=0.1, y=1, orientation="v"))
fig=go.Figure(data,layout=layout)
fig.show()
                 


# The growth in contents are higher in the first three months and the last three months of the year.
# 
# Least number of contents are added in the month of February.

# # Genre Relationship:

# ### 1. Movies Genre:

# In[21]:


df.head()


# In[22]:


from sklearn.preprocessing import MultiLabelBinarizer

def relation_heatmap(df,title):
    df['genre']=df['listed_in'].apply(lambda x:x.replace(' ,',',').replace(', ',',').split(','))
    Types=[]
    for i in df['genre']:
        Types+=i
    Types=set(Types)
    print("There are {} types in the Netflix {} Dataset".format(len(Types),title))  
    test=df['genre']
    mlb=MultiLabelBinarizer()
    res=pd.DataFrame(mlb.fit_transform(test),columns=mlb.classes_,index=test.index)
    corr=res.corr()
    mask=np.zeros_like(corr,dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    fig,ax=plt.subplots(figsize=(10,7))
    pl=sns.heatmap(corr,cmap='coolwarm',mask=mask,vmax=0.5, vmin=-.5, center=0, square=True, linewidths=.7,
                     cbar_kws={"shrink": 0.6})
    plt.show()


# In[23]:


relation_heatmap(df_movies,'Movie')


# The negative relationship between **drama and documentary** is remarkable.
# 
# We can see that there are many **dramas** for **independent and international films.**
# 
# And also **Sci-Fi & Fantasy for action & Adventure** and **horror movies for thrillers**
# 
# 

# ### 2. TV_shows genre:

# In[24]:


relation_heatmap(df_tv, 'TV Show')


# TV shows are **more clearly correlated** than movies.
# 
# The **negative** relationship between **kid's TV and International Tv Shows** is remarkable.
# 
# There is a strong **positive** corelation between **Science & Natural and Docuseries.**

# ## Distribution of Movie Duration:

# In[29]:


from scipy.stats import norm

plt.figure(figsize=(15,7))
sns.distplot(df_movies['duration'].str.extract('(\d+)'),fit=norm,kde=False,color=['red'])
plt.title('Distplot with Normal distribution for Movies',fontweight='bold')
plt.show()


# Form the above plot we can say that majority of the movies have duration ranging from **85 min to 120 min.**

# ## Distribution for TV_shows:

# In[41]:


plt.figure(figsize=(15,7))
ax = sns.countplot(df_tv['duration'],order = df_tv['duration'].value_counts().index,palette="tab10")
plt.title('Countplot for Seasons in TV_Shows',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))

plt.figure(figsize=(15,7))
ax = sns.barplot(x=((df_tv['duration'].value_counts()/df_tv.shape[0])*100).index,
                 y=round(((df_tv['duration'].value_counts()/df_tv.shape[0])*100),2).values,
                 palette="tab10")
plt.title('Percentage of Seasons in TV_Shows',fontweight="bold")
plt.xticks(rotation=90)
for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, (p.get_height() * 1.005)))
plt.show()


# From the above plots we can say that 90% of the TV_Shows end by at most Season 3.

# ## Top10 Genre in Movies and TV Shows:

# In[68]:


plt.figure(figsize=(15,5))
sns.barplot(x = df_movies["listed_in"].value_counts().head(10).index,
            y = df_movies["listed_in"].value_counts().head(10).values,palette="viridis")
plt.xticks(rotation=90)
plt.title("Top10 Genre in Movies",fontweight="bold")
plt.show()


# In[67]:


plt.figure(figsize=(15,5))
sns.barplot(x = df_tv["listed_in"].value_counts().head(10).index,
            y = df_tv["listed_in"].value_counts().head(10).values,palette="rocket")
plt.xticks(rotation=90)
plt.title("Top10 Genre in TV Shows",fontweight="bold")
plt.show()


# ## Top 20 artist present on Netflix:

# In[78]:


df['cast_name'] = df['cast'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 
cast_count = []
for i in df['cast_name']: 
    cast_count += i
    
cast_dict = dict((i, cast_count.count(i)) for i in cast_count)

df_cast_count = pd.DataFrame(cast_dict.values(),cast_dict.keys()).reset_index().sort_values(0,ascending=False).rename(
    columns = {'index' : 'cast_name', 0 : 'count'}).iloc[1:21]


# In[50]:


plt.figure(figsize=(15,5))
sns.barplot(x='cast_name',y='count',data=df_cast_count,palette="mako")
plt.title("Top20 Artist on Netflix",fontweight="bold")
plt.xticks(rotation=90)
plt.show()


# In[ ]:




