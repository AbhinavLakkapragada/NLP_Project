#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import warnings
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)

warnings.filterwarnings('ignore')


# In[2]:


col_names = ['ID', 'Entity', 'Sentiment', 'Content']


# In[3]:


train = pd.read_csv(r"C:\Users\Abhinav Lakkapragada\Downloads\twitter_training.csv\twitter_training.csv",names=col_names)
train.sample(5)


# In[4]:


test = pd.read_csv(r"C:\Users\Abhinav Lakkapragada\Downloads\twitter_validation.csv",names=col_names)
test.sample(5)


# In[5]:


train.shape


# In[6]:


# Data Preprocessing

train.info()


# In[7]:


# checking values
train.isnull().sum()


# In[8]:


train.dropna(subset=['Content'], inplace=True)


# In[9]:


# checking duplicate values
train.duplicated().sum()


# In[10]:


# replacing irrelavant with neutral
train['Sentiment'] = train['Sentiment'].replace('Irrelevant', 'Neutral')
test['Sentiment'] = test['Sentiment'].replace('Irrelevant', 'Neutral')


# In[11]:


train.head()


# In[12]:


# dropping duplicates
train = train.drop_duplicates(keep='first')


# In[13]:


train.duplicated().sum()


# In[14]:


train.shape


# In[15]:


train.head()


# In[16]:


# Exploratory Data Analysis (EDA)

train['Sentiment'].value_counts()


# In[17]:


plt.pie(train['Sentiment'].value_counts(), labels=['Neutral','Negative','Positive'],autopct='%0.2f',colors=['yellow','red','green'])
plt.show()


# In[18]:


# Data is imbalanced
# hence we will give more importance to accuracy than precision


# In[19]:


nltk.download('punkt')


# In[20]:


train['num_char'] = train['Content'].apply(len) # no of characters of each text


# In[21]:


train.head()


# In[22]:


# number of words
train['num_words'] = train['Content'].apply(lambda x: len(nltk.word_tokenize(x)))


# In[23]:


train.head()


# In[24]:


# number of sentences
train['num_sentences'] = train['Content'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[25]:


train.head()


# In[26]:


# data description
train[['num_char','num_words','num_sentences']].describe()


# In[27]:


plt.figure(figsize=(6,4))
sns.histplot(train[train['Sentiment'] == 'Neutral']['num_char'])
sns.histplot(train[train['Sentiment'] == 'Negative']['num_char'], color='red')
sns.histplot(train[train['Sentiment'] == 'Positive']['num_char'], color='green')


# In[28]:


plt.figure(figsize=(6,4))
sns.histplot(train[train['Sentiment'] == 'Neutral']['num_words'])
sns.histplot(train[train['Sentiment'] == 'Negative']['num_words'], color='red')
sns.histplot(train[train['Sentiment'] == 'Positive']['num_words'], color='green')


# In[29]:


df=train.drop(['ID','Entity'],axis=1)


# In[30]:


df.head()


# In[31]:


# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Sentiment'],drop_first=True)
df_encoded['Sentiment_Neutral'] = df_encoded['Sentiment_Neutral'].astype(int)
df_encoded['Sentiment_Positive'] = df_encoded['Sentiment_Positive'].astype(int)


# In[32]:


df_encoded=df_encoded.drop(['Content'],axis=1)


# In[33]:


sns.heatmap(df_encoded.corr(),annot=True)


# In[34]:


# Text Preprocessing
# Except for removing stopwords, we are doing the rest

import string # for punctuation
# string.punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:] # list is mutable, so you have to do cloning, else if you clear y, text gets cleared too
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[35]:


# Perform Label encoding
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])


# In[36]:


df.head()


# In[37]:


df['transformed_text'] = df['Content'].apply(transform_text)


# In[38]:


df.head()


# In[39]:


from wordcloud import WordCloud
wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='black')


# In[40]:


condition = (df['sentiment_encoded']==0)

# Generate the word cloud for "transformed_text" when both conditions are met
wcNeutral = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[41]:


plt.figure(figsize=(8,8))
plt.imshow(wcNeutral)


# In[42]:


condition = (df['sentiment_encoded']==1)

# Generate the word cloud for "transformed_text" when both conditions are met
wcNegative = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[43]:


plt.figure(figsize=(8,8))
plt.imshow(wcNegative)


# In[44]:


condition = (df['sentiment_encoded']==2)

# Generate the word cloud for "transformed_text" when both conditions are met
wcPositive = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[45]:


plt.figure(figsize=(8,8))
plt.imshow(wcPositive)


# In[46]:


df=df.drop(['Sentiment'],axis=1)


# In[47]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[48]:


print(X.shape)


# In[49]:


y = df['sentiment_encoded'].values


# In[50]:


print(y.shape)


# In[51]:


from sklearn.model_selection import train_test_split


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[53]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

bnb = BernoulliNB()


# In[54]:


# TFIDF Vectorizer - BernoulliNB
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
print(precision_score(y_test,y_pred,average=None))


# In[55]:


from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV

# Create a pipeline with just the logistic regression classifier
pipeline = Pipeline([
    ('lr_clf', LogisticRegression(solver='liblinear'))
])

# defining the hyperparameter grid for logistic regression (C parameter)
params = {'lr_clf__C': [1, 5, 10]}

grid_cv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_pipe.fit(X_train, y_train)
print('Optimized Hyperparameters:', grid_cv_pipe.best_params_)

pred = grid_cv_pipe.predict(X_test)
print('Optimized Accuracy Score: {0: .3f}'.format(accuracy_score(y_test, pred)))


# In[56]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[61]:


# Now we repeat it while removing the stopwords


# In[68]:


col_names = ['ID', 'Entity', 'Sentiment', 'Content']


# In[69]:


train1 = pd.read_csv(r"C:\Users\Abhinav Lakkapragada\Downloads\twitter_training.csv\twitter_training.csv",names=col_names)
train1.sample(5)


# In[70]:


train = pd.read_csv(r"C:\Users\Abhinav Lakkapragada\Downloads\twitter_training.csv\twitter_training.csv",names=col_names)
train.sample(5)


# In[71]:


test = pd.read_csv(r"C:\Users\Abhinav Lakkapragada\Downloads\twitter_validation.csv",names=col_names)
test.sample(5)


# In[72]:


train.shape


# In[73]:


# Data Preprocessing

train.info()


# In[74]:


# checking values
train.isnull().sum()


# In[75]:


train.dropna(subset=['Content'], inplace=True)


# In[76]:


# checking duplicate values
train.duplicated().sum()


# In[77]:


# replacing irrelavant with neutral
train['Sentiment'] = train['Sentiment'].replace('Irrelevant', 'Neutral')
test['Sentiment'] = test['Sentiment'].replace('Irrelevant', 'Neutral')


# In[78]:


train.head()


# In[79]:


# dropping duplicates
train = train.drop_duplicates(keep='first')


# In[80]:


train.duplicated().sum()


# In[81]:


train.shape


# In[82]:


train.head()


# In[83]:


# Exploratory Data Analysis (EDA)

train['Sentiment'].value_counts()


# In[84]:


plt.pie(train['Sentiment'].value_counts(), labels=['Neutral','Negative','Positive'],autopct='%0.2f',colors=['yellow','red','green'])
plt.show()


# In[85]:


# Data is imbalanced
# hence we will give more importance to accuracy than precision


# In[86]:


nltk.download('punkt')


# In[87]:


train['num_char'] = train['Content'].apply(len) # no of characters of each text


# In[88]:


train.head()


# In[89]:


# number of words
train['num_words'] = train['Content'].apply(lambda x: len(nltk.word_tokenize(x)))


# In[90]:


train.head()


# In[91]:


# number of sentences
train['num_sentences'] = train['Content'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[92]:


train.head()


# In[93]:


# data description
train[['num_char','num_words','num_sentences']].describe()


# In[94]:


plt.figure(figsize=(6,4))
sns.histplot(train[train['Sentiment'] == 'Neutral']['num_char'])
sns.histplot(train[train['Sentiment'] == 'Negative']['num_char'], color='red')
sns.histplot(train[train['Sentiment'] == 'Positive']['num_char'], color='green')


# In[95]:


plt.figure(figsize=(6,4))
sns.histplot(train[train['Sentiment'] == 'Neutral']['num_words'])
sns.histplot(train[train['Sentiment'] == 'Negative']['num_words'], color='red')
sns.histplot(train[train['Sentiment'] == 'Positive']['num_words'], color='green')


# In[96]:


df=train.drop(['ID','Entity'],axis=1)


# In[97]:


# Perform one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Sentiment'],drop_first=True)
df_encoded['Sentiment_Neutral'] = df_encoded['Sentiment_Neutral'].astype(int)
df_encoded['Sentiment_Positive'] = df_encoded['Sentiment_Positive'].astype(int)


# In[98]:


df_encoded=df_encoded.drop(['Content'],axis=1)


# In[99]:


sns.heatmap(df_encoded.corr(),annot=True)


# In[100]:


# Text Preprocessing

from nltk.corpus import stopwords
import string # for punctuation
# string.punctuation
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:] # list is mutable, so you have to do cloning, else if you clear y, text gets cleared too
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[101]:


# Perform Label encoding
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])


# In[102]:


df.head()


# In[103]:


df['transformed_text'] = df['Content'].apply(transform_text)


# In[104]:


df.head()


# In[105]:


from wordcloud import WordCloud
wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='black')


# In[106]:


condition = (df['sentiment_encoded']==0)

# Generate the word cloud for "transformed_text" when both conditions are met
wcNeutral = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[107]:


plt.figure(figsize=(8,8))
plt.imshow(wcNeutral)


# In[108]:


condition = (df['sentiment_encoded']==1)

# Generate the word cloud for "transformed_text" when both conditions are met
wcNegative = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[109]:


plt.figure(figsize=(8,8))
plt.imshow(wcNegative)


# In[110]:


condition = (df['sentiment_encoded']==2)

# Generate the word cloud for "transformed_text" when both conditions are met
wcPositive = WordCloud().generate(df[condition]['transformed_text'].str.cat(sep=" "))


# In[111]:


plt.figure(figsize=(8,8))
plt.imshow(wcPositive)


# In[112]:


df=df.drop(['Sentiment'],axis=1)


# In[113]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[114]:


print(X.shape)


# In[115]:


y = df['sentiment_encoded'].values


# In[116]:


print(y.shape)


# In[117]:


from sklearn.model_selection import train_test_split


# In[118]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X,y,test_size=0.2,random_state=2)


# In[119]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

bnb = BernoulliNB()


# In[123]:


# TFIDF Vectorizer - BernoulliNB
bnb.fit(X_train1, y_train1)
y_pred1 = bnb.predict(X_test1)
print(precision_score(y_test1,y_pred1,average=None))


# In[125]:


from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV

# Create a pipeline with just the logistic regression classifier
pipeline = Pipeline([
    ('lr_clf', LogisticRegression(solver='liblinear'))
])

# defining the hyperparameter grid for logistic regression (C parameter)
params = {'lr_clf__C': [1, 5, 10]}

grid_cv_pipe = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='accuracy', verbose=1)
grid_cv_pipe.fit(X_train1, y_train1)
print('Optimized Hyperparameters:', grid_cv_pipe.best_params_)

pred1 = grid_cv_pipe.predict(X_test1)
print('Optimized Accuracy Score: {0: .3f}'.format(accuracy_score(y_test1, pred1)))


# In[126]:


from sklearn.metrics import classification_report
print(classification_report(y_test1,y_pred1))


# In[ ]:




