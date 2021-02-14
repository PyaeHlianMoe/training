import pandas as pd


def to_full_class_name(sentiment):
    sentiment = str(sentiment)
    if sentiment == "pos":
        return "positive"
    return "negative"


# TODO: To replace with correct path
train = pd.read_csv('drive/MyDrive/FYP_PyaeHlianMoe/data/yelp/train.csv')
test = pd.read_csv('drive/MyDrive/FYP_PyaeHlianMoe/data/yelp/test.csv')
print(train.shape,test.shape)

test["full_sentiment"] = test.Sentiment.apply(to_full_class_name)
train["full_sentiment"] = train.Sentiment.apply(to_full_class_name)
test.head()
train.head()
