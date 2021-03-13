import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import random
import re
from nltk.stem.wordnet import WordNetLemmatizer
import string
import tweepy
import pandas as pd
import numpy as np


# loading data
twitter_sample = twitter_samples.fileids()
# print(twitter_sample)

# getting positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

print(len(positive_tweets))
print(len(negative_tweets))

# removing emojis

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)


positive_noemo = []
negative_noemo = []

for tweets in positive_tweets:
    positive_noemo.append(emoji_pattern.sub(r'', tweets))

for tweets in negative_tweets:
    negative_noemo.append(emoji_pattern.sub(r'', tweets))


# tokenization
tweet_tokenizer = TweetTokenizer()
positive_tokenized = [tweet_tokenizer.tokenize(tweet) for tweet in positive_noemo]
negative_tokenized = [tweet_tokenizer.tokenize(tweet) for tweet in negative_noemo]
print('positive tokens:', positive_tokenized[:10])


# filtering tokens
def filter_tokens(tweet_tokens, stopwords=()):
    filtered_tokens = []
    for token in tweet_tokens:
        token = token.lower()
        token = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            ' ', token)
        token = re.sub(r'[^A-Za-z0-9 ]+', '', token)


        tweet_lemming = WordNetLemmatizer()
        token = tweet_lemming.lemmatize(token)  # word same as token

        if len(token) > 0 and token not in string.punctuation and token not in stopwords:
            filtered_tokens.append(token)

    return filtered_tokens


positive_filtered = []
negative_filtered = []

for tokens in positive_tokenized:
    positive_filtered.append(filter_tokens(tokens))

for tokens in negative_tokenized:
    negative_filtered.append(filter_tokens(tokens))

print('positive filtered: ', positive_filtered[:10])
print('negative filtered: ', negative_filtered[:10])

positive_data = [(tweet, 'Positive') for tweet in positive_filtered]
negative_data = [(tweet, 'Negative') for tweet in negative_filtered]
print('positve data: ', positive_data[:10])


dataset = positive_data + negative_data
random.shuffle(dataset)
print(dataset[:20])

dataframe = pd.DataFrame(dataset,columns=['tweet','label'])
print(dataframe.head())
df_copy = dataframe.copy()

dataframe['label'].replace({"Negative": 0, "Positive": 1}, inplace=True)


def try_join(token_list):
    try:
        return ','.join(map(str, token_list))
    except TypeError:
        return np.nan


dataframe['tweet'] = [try_join(token_list) for token_list in dataframe['tweet']]

train_df = dataframe[:7000]
test_df = dataframe['tweet'][7000:]
print(train_df.head())
print(test_df.head())


# print(train_df.info())

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english',lowercase=False)
bag_of_words = vectorizer.fit_transform(dataframe['tweet'])
print(bag_of_words)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bag_of_words[:7000]
test_bow = bag_of_words[7000:]
# print('test df')
print('bow')
print(train_bow)
print(test_bow)


# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train_df['label'], random_state=42, test_size=0.3)
#
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
#
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.5 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

print(f1_score(yvalid, prediction_int)) # calculating f1 score
#

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.5
test_pred_int = test_pred_int.astype(np.int)
test_df['label'] = test_pred_int

y_pred = lreg.predict(xvalid_bow)

from sklearn import metrics
print("Logistic Regression accuracy (in %):", metrics.accuracy_score(yvalid, y_pred) * 100)
print(lreg.score(xvalid_bow,yvalid))
print(lreg.score(xtrain_bow,ytrain))

# from sklearn.naive_bayes import GaussianNB
#
# gnb = GaussianNB()
# gnb.fit(xtrain_bow, ytrain)
#
# y_pred = gnb.predict(xvalid_bow)
#
# from sklearn import metrics
# print("Gaussian Naive Bayes accuracy(in %):", metrics.accuracy_score(yvalid, y_pred) * 100)

# from sklearn import linear_model
# lasso_reg = linear_model.Ridge(alpha=50,max_iter=100,tol=0.1)
# lasso_reg.fit(xtrain_bow,ytrain)
#
# print(lasso_reg.score(xvalid_bow,yvalid))
# print(lasso_reg.score(xtrain_bow,ytrain))


import pickle
from sklearn.pipeline import Pipeline
xtrain_bow = train_df['tweet']
ytrain = train_df['label']

# build the pipeline
pipe = Pipeline([('vect', CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english',lowercase=False)),
                 ('model', LogisticRegression())])




model = pipe.fit(xtrain_bow, ytrain)

# export model
with open('model.pkl', 'wb') as model_file:
  pickle.dump(model, model_file)



























































































# token = '1337701516203806722-Tc0qgs0oxB0oOqK1CMtXt4u3NmvqOu'
# token_secret = 'BYjtc1Yfi5WJkM8ODHNiH4pSZtfxB49C4ELzJ2pOMTlm0'
# consumer_key = 'EPvgyr0BY3sfJQex4qyKGMGUW'
# consumer_secret = '1XwRpfw2YZOat9gD9OLf8RC5qw0HqxUL0Dvp3QmQ33ffoJlTiD'
#
# auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# auth.set_access_token(token, token_secret)
# api = tweepy.API(auth)

#
# def get_tweets(keyword):  # keyword can be username or hash tag
#
#     number_of_tweets = 100
#     twitter_tweets = api.search(q=keyword, count=number_of_tweets, lang='eng')
#
#     return twitter_tweets
#
#
# def get_user_tweets(username):  # gets tweets from users timeline
#     number_of_tweets = 100
#     user_tweets = api.user_timeline(screen_name=username, count=number_of_tweets)
#
#     return user_tweets
#
#
# def create_test_data(twitter_tweets):  # preparing test data
#     return [{'tweet': tweet.text, 'label': None} for tweet in twitter_tweets]
#
#
# # # user_tweets = get_user_tweets('@Thom_astro')
#
# tweets = get_tweets('#joy')
# for t in tweets:
#     print('\nNew Tweet\n',t)
