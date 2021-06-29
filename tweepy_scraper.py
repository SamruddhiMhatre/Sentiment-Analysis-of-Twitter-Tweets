import tweepy
import json
import pandas as pd
import time

token = ''
token_secret = ''
consumer_key = ''
consumer_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(token, token_secret)
api = tweepy.API(auth)


search_hashtags = []


def get_tweets(search_hashtags):

    max_tweets = 50000
    date_since = '2019-01-01'
    my_list_of_dicts = []
    try:
        for query in search_hashtags:
            tweets = [status for status in
                      tweepy.Cursor(api.search, q=query, lang="en", geocode='9.0845755,8.674252499999994,550km',
                                    since=date_since, rpp=100).items(max_tweets)]

            print(len(tweets))
            for each_json_tweet in tweets:
                my_list_of_dicts.append(each_json_tweet._json)
    except:
        time.sleep(3)

    with open('tweet_json_Data.txt', 'w') as file:
        file.write(json.dumps(my_list_of_dicts, indent=4))

    my_demo_list = []
    with open('tweet_json_Data.txt', encoding='utf-8') as json_file:
        all_data = json.load(json_file)

        for each_dictionary in all_data:
            print(each_dictionary)
            tweet_id = each_dictionary['id']
            text = each_dictionary['text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            geo = each_dictionary['geo']
            coordinates = each_dictionary['coordinates']
            place = each_dictionary['place']
            lang = each_dictionary['lang']

            my_demo_list.append({'tweet_id': str(tweet_id),
                                 'text': str(text),
                                 'favorite_count': int(favorite_count),
                                 'retweet_count': int(retweet_count),
                                 'created_at': created_at,
                                 'geo': geo,
                                 'coordinates': coordinates,
                                 'place': place,
                                 'lang': lang
                                 })

            tweet_dataset = pd.DataFrame(my_demo_list, columns=['tweet_id', 'text',
                                                                'favorite_count', 'retweet_count',
                                                                'created_at', 'geo', 'coordinates', 'place', 'lang'])

    # print(tweet_dataset.head())
    # tweet_dataset.to_csv('tweet_data.csv')

    return tweet_dataset.to_csv('tweets_data.csv')def get_tweets(keyword):

    max_tweets = 1000
    date_since = '2019-01-01'
    my_list_of_dicts = []
    try:
        tweets = [status for status in
                  tweepy.Cursor(api.search, q=keyword, lang="en", geocode='9.0845755,8.674252499999994,550km',
                                    since=date_since, rpp=100).items(max_tweets)]

        # print(len(tweets))
        for each_json_tweet in tweets:
            my_list_of_dicts.append(each_json_tweet._json)
    except:
        time.sleep(3)

    with open('tweet_json_Data.txt', 'w') as file:
        file.write(json.dumps(my_list_of_dicts, indent=4))

    my_demo_list = []
    with open('tweet_json_Data.txt', encoding='utf-8') as json_file:
        all_data = json.load(json_file)

        for each_dictionary in all_data:
            print(each_dictionary)
            tweet_id = each_dictionary['id']
            text = each_dictionary['text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            geo = each_dictionary['geo']
            coordinates = each_dictionary['coordinates']
            place = each_dictionary['place']
            lang = each_dictionary['lang']

            my_demo_list.append({'tweet_id': str(tweet_id),
                                 'text': str(text),
                                 'favorite_count': int(favorite_count),
                                 'retweet_count': int(retweet_count),
                                 'created_at': created_at,
                                 'geo': geo,
                                 'coordinates': coordinates,
                                 'place': place,
                                 'lang': lang
                                 })

            tweet_dataset = pd.DataFrame(my_demo_list, columns=['tweet_id', 'text',
                                                                'favorite_count', 'retweet_count',
                                                                'created_at', 'geo', 'coordinates', 'place', 'lang'])

    # print(tweet_dataset.head())
    # tweet_dataset.to_csv('tweet_data.csv')

    return tweet_dataset




def get_user_tweets(username):  # gets tweets from users timeline
    number_of_tweets = 1000
    my_list_of_dicts = []
    user_tweets = api.user_timeline(screen_name=username, count=number_of_tweets)

    for each_json_tweet in user_tweets:
        my_list_of_dicts.append(each_json_tweet._json)

    with open('tweet_json_Data.txt', 'w') as file:
        file.write(json.dumps(my_list_of_dicts, indent=4))

    my_demo_list = []
    with open('tweet_json_Data.txt', encoding='utf-8') as json_file:
        all_data = json.load(json_file)

        for each_dictionary in all_data:
            print(each_dictionary)
            tweet_id = each_dictionary['id']
            text = each_dictionary['text']
            favorite_count = each_dictionary['favorite_count']
            retweet_count = each_dictionary['retweet_count']
            created_at = each_dictionary['created_at']
            geo = each_dictionary['geo']
            coordinates = each_dictionary['coordinates']
            place = each_dictionary['place']
            lang = each_dictionary['lang']

            my_demo_list.append({'tweet_id': str(tweet_id),
                                 'text': str(text),
                                 'favorite_count': int(favorite_count),
                                 'retweet_count': int(retweet_count),
                                 'created_at': created_at,
                                 'geo': geo,
                                 'coordinates': coordinates,
                                 'place': place,
                                 'lang': lang
                                 })

            user_tweets_df = pd.DataFrame(my_demo_list, columns=['tweet_id', 'text',
                                                                'favorite_count', 'retweet_count',
                                                                'created_at', 'geo', 'coordinates', 'place', 'lang'])


    return user_tweets_df


