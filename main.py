import nltk

from clean_tweet import textFiltered, textTokenized
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
import random
import re
from nltk.stem.wordnet import WordNetLemmatizer
import string
import tweepy
import csv
import time
import json
import pandas as pd
from nltk import FreqDist
from nltk.tokenize import word_tokenize

twitter_sample = twitter_samples.fileids()
# print(twitter_sample)

# getting positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# tokenization
tweet_tokenizer = TweetTokenizer()
positive_tokenized = [tweet_tokenizer.tokenize(tweet) for tweet in positive_tweets]
negative_tokenized = [tweet_tokenizer.tokenize(tweet) for tweet in negative_tweets]
print('positive tokens:', positive_tokenized[:10])


# filtering tokens
def filter_tokens(tokens, stopwords=()):
    filtered_tokens = []
    for token in tokens:
        token = token.lower()
        token = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            ' ', token)
        token = re.sub('(@[A-Za-z0-9_]+)', '', token)

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token)  # word same as token

        if len(token) > 0 and token not in string.punctuation and token not in stopwords:
            filtered_tokens.append(token)

    return filtered_tokens


positive_filtered = []
negative_filtered = []

for tokens in positive_tokenized:
    positive_filtered.append(filter_tokens(tokens))

for tokens in negative_tokenized:
    negative_filtered.append(filter_tokens(tokens))

print('positive filtered: ',positive_filtered[:10])
print('negative filtered: ',negative_filtered[:10])
#
# for fd in positive_filtered:
#     fd = nltk.FreqDist([w for w in fd])
# print(fd['top'])




# data creation
def model_data(filtered):
    for tokens in filtered:
        yield dict([token, True] for token in tokens)


positive_model_dict = model_data(positive_filtered)
negative_model_dict = model_data(negative_filtered)

positive_data = [(tweet, 'Positive') for tweet in positive_model_dict]
negative_data = [(tweet, 'Negative') for tweet in negative_model_dict]
print('positve data: ',positive_data[:10])

dataset = positive_data + negative_data
random.shuffle(dataset)
print(dataset[:20])


train_data = dataset[: round(len(dataset) * 0.7)]
test_data = dataset[round(len(dataset) * 0.7):]

from nltk import classify
from nltk import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(train_data)
print("Training accuracy is:{}\n".format(classify.accuracy(classifier, train_data)))
print("Testing accuracy is:{}\n".format(classify.accuracy(classifier, test_data)))
print(classifier.show_most_informative_features(10))
# print(ensemble_clf.classify(feats), ensemble_clf.confidence(feats))

test_tweet = "If I could give less than one star, that would have been my choice.  I rent a home and Per my lease agreement it is MY responsibility to pay their Pool Service company.  Within the last year they changed to PoolServ.  I have had  major issues with new techs every week, never checking PH balances, cleaning the filter, and not showing up at all 2 weeks in the past 2 months. I have had 4 different techs in the past 4 weeks.   I have emailed and called them and they never respond back nor even acknowledged my concerns or requests.  I cannot change companies but I'm required to still pay for lousy or no service.  Attached are a couple pictures of my pool recently due to one tech just didn't put any chlorine in it at all according to the tech who came the following week to attempt to clean it up.  Please think twice before working with these people.  No one wants to work with a business that doesn't return phone calls or emails."
test_token = tweet_tokenizer.tokenize(test_tweet)
test_filtered = filter_tokens(test_token)

from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(test_tweet))
print(sia.polarity_scores(test_tweet)['pos'])

print(test_tweet)
print(classifier.classify(dict([token, True] for token in test_filtered)))


token = '1337701516203806722-Tc0qgs0oxB0oOqK1CMtXt4u3NmvqOu'
token_secret = 'BYjtc1Yfi5WJkM8ODHNiH4pSZtfxB49C4ELzJ2pOMTlm0'
consumer_key = 'EPvgyr0BY3sfJQex4qyKGMGUW'
consumer_secret = '1XwRpfw2YZOat9gD9OLf8RC5qw0HqxUL0Dvp3QmQ33ffoJlTiD'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)


def get_tweets(keyword):  # keyword can be username or hash tag

    number_of_tweets = 100
    tweets = api.search(q=keyword, count=number_of_tweets, lang='eng')

    return tweets


def get_user_tweets(username):  # gets tweets from users timeline
    number_of_tweets = 100
    tweets = api.user_timeline(screen_name=username, count=number_of_tweets)

    return tweets


def create_test_data(tweets):  # preparing test data
    return [{'tweet': tweet.text, 'lable': None} for tweet in tweets]


# user_tweets = get_user_tweets('@Thom_astro')
# tweets = get_tweets('#covid19')
# print(tweets)
query = 'covid19'
max_tweets = 2000
tweets = [status for status in tweepy.Cursor(api.search, q=query).items(max_tweets)]

my_list_of_dicts = []
for each_json_tweet in tweets:
    my_list_of_dicts.append(each_json_tweet._json)

with open('tweet_json_Data.txt', 'w') as file:
    file.write(json.dumps(my_list_of_dicts, indent=4))

my_demo_list = []
with open('tweet_json_Data.txt', encoding='utf-8') as json_file:
    all_data = json.load(json_file)
    for each_dictionary in all_data:
        tweet_id = each_dictionary['id']
        text = each_dictionary['text']
        favorite_count = each_dictionary['favorite_count']
        retweet_count = each_dictionary['retweet_count']
        created_at = each_dictionary['created_at']
        my_demo_list.append({'tweet_id': str(tweet_id),
                             'text': str(text),
                             'favorite_count': int(favorite_count),
                             'retweet_count': int(retweet_count),
                             'created_at': created_at,
                             })

        tweet_dataset = pd.DataFrame(my_demo_list, columns=
        ['tweet_id', 'text',
         'favorite_count', 'retweet_count',
         'created_at'])

# Writing tweet dataset ti csv file for future reference

sentiment = []
sentiment_pos_score = []
sentiment_neg_score = []
for val in tweet_dataset['text']:
    test_token = tweet_tokenizer.tokenize(val)
    test_filtered = filter_tokens(test_token)

    print(val)
    print(classifier.classify(dict([token, True] for token in test_filtered)))
    sentiment.append(classifier.classify(dict([token, True] for token in test_filtered)))
    print(sia.polarity_scores(val))
    sentiment_pos_score.append(sia.polarity_scores(val)['pos'])
    sentiment_neg_score.append(sia.polarity_scores(val)['neg'])
print(sentiment)
tweet_dataset['sentiment'] = sentiment
tweet_dataset['pos_score'] = sentiment_pos_score
tweet_dataset['neg_score'] = sentiment_neg_score

print(tweet_dataset[['text','sentiment','pos_score','neg_score']])
tweet_dataset.to_csv('tweet_data.csv')
