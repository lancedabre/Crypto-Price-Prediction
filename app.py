"""Entry point for the crypto price prediction project."""

from nlp_pipeline import get_daily_sentiment


def main():
    rss_feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]
    sentiment_df = get_daily_sentiment(rss_feeds)
    print(sentiment_df.head())


if __name__ == "__main__":
    main()
