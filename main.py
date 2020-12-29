import tweepy
import csv
import time

token = '1337701516203806722-Tc0qgs0oxB0oOqK1CMtXt4u3NmvqOu'
token_secret = 'BYjtc1Yfi5WJkM8ODHNiH4pSZtfxB49C4ELzJ2pOMTlm0'
consumer_key = 'EPvgyr0BY3sfJQex4qyKGMGUW'
consumer_secret = '1XwRpfw2YZOat9gD9OLf8RC5qw0HqxUL0Dvp3QmQ33ffoJlTiD'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token, token_secret)

corpus_csv = '/Users/samruddhimhatre/PycharmProjects/SentimentAnalysis/corpus.csv'
train_files = '/Users/samruddhimhatre/PycharmProjects/SentimentAnalysis/train_files.csv'


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


def createLimitedTrainingCorpus(corpusFile, tweetDataFile):
    import csv
    corpus = []
    with open(corpusFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        for row in lineReader:
            corpus.append({"tweet_id": row[2], "label": row[1], "topic": row[0]})

    trainingData = []

    for label in ["positive", "negative"]:
        i = 1
        for tweet in corpus:
            if tweet["label"] == label and i <= 50:
                # print(tweet["label"])
                try:
                    status = api.get_status(tweet["tweet_id"])
                    # Returns a twitter.Status object
                    print("Tweet fetched" + status.text)
                    tweet["text"] = status.text
                    # tweet is a dictionary which already has tweet_id and label (positive/negative/neutral)
                    # Add another attribute now, the tweet text
                    trainingData.append(tweet)
                    i = i + 1
                except Exception as e:
                    print(e)

    # Once the tweets are downloaded write them to a csv, so you won't have to wait 10 hours
    # every time you run this code :)
    with open(tweetDataFile, 'wb') as csvfile:
        linewriter = csv.writer(csvfile, delimiter=',', quotechar="\"")
        # We'll add a try catch block here so that we still get the training data even if the write
        # fails
        for tweet in trainingData:
            try:
                linewriter.writerow([tweet["tweet_id"], tweet["text"], tweet["label"], tweet["topic"]])
            except Exception as e:
                print(e)
    return trainingData


user_tweets = get_user_tweets('@Thom_astro')
tweets = get_tweets('#covid19')

# test_data = create_test_data(tweets)
train_data = createLimitedTrainingCorpus(corpus_csv, train_files)
print(train_data[98])
