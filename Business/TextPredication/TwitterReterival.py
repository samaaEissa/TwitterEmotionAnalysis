#pip install python-twitter
import twitter

CONSUMER_KEY=''
CONSUMER_SECRET=''
ACCESS_TOKEN_KEY=''
ACCESS_TOKEN_SECRET=''
api = twitter.Api(CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)


#######################################################
def get_tweetsbyKeyword(term):
    Tweets=[]
    tweets=api.GetSearch(term,count=100,lang="ar")
    for tweet in tweets:
        Tweets.append(tweet.text)
    return Tweets
#---------------------------------------------

def get_tweetsbyAccount(accountName):
    Tweets=[]
    timeline = api.GetUserTimeline(screen_name=accountName, count=50)
    earliest_tweet = min(timeline, key=lambda x: x.id).id
    print("getting tweets before:", earliest_tweet)
    tweets = api.GetUserTimeline(
            screen_name=accountName, max_id=earliest_tweet, count=50
        )
    timeline=tweets
    # while True:
    #     tweets = api.GetUserTimeline(
    #         screen_name=screen_name, max_id=earliest_tweet, count=10
    #     )
    #     new_earliest = min(tweets, key=lambda x: x.id).id

    #     if not tweets or new_earliest == earliest_tweet:
    #         break
    #     else:
    #         earliest_tweet = new_earliest
    #         print("getting tweets before:", earliest_tweet)
    #         timeline += tweets
            
    for tweet in timeline:
        Tweets.append(tweet._json['text'])       
    return Tweets
####################################################    
# screen_name = 'VodafoneEgypt'
# tweets = get_tweets(screen_name)
# for tweet in tweets:
#     print(tweet)
#     print('\n')
  