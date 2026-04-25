import datetime
from typing import List, Dict

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


DEFAULT_RSS_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",  # CoinDesk news RSS
    "https://cointelegraph.com/rss",  # Cointelegraph news RSS
]


def _load_sentiment_analyzer(custom_lexicon: Dict[str, float] = None) -> SentimentIntensityAnalyzer:
    """Create a VADER analyzer and optionally extend its lexicon.

    Args:
        custom_lexicon: Optional dictionary of token -> sentiment score.
            Use this to add crypto-specific terms like 'HODL' or 'bullish'.
            Example: {'HODL': 2.0, 'bullish': 2.5}

    Returns:
        SentimentIntensityAnalyzer: configured analyzer.
    """
    analyzer = SentimentIntensityAnalyzer()
    if custom_lexicon:
        analyzer.lexicon.update(custom_lexicon)
    return analyzer


def _parse_headlines_from_feeds(rss_urls: List[str]) -> List[Dict[str, str]]:
    """Fetch headlines from a list of RSS feed URLs."""
    headlines = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = getattr(entry, "title", "")
            published = getattr(entry, "published", None) or getattr(entry, "updated", None)
            if not title or not published:
                continue
            headlines.append({"title": title, "published": published})
    return headlines


def _normalize_date(date_text: str) -> datetime.date:
    """Normalize RSS published date text into a date object."""
    parsed = feedparser.parse(date_text)
    if parsed.bozo and not parsed.entries:
        # Fall back to common formats if parsing fails
        try:
            return datetime.datetime.strptime(date_text, "%Y-%m-%d").date()
        except ValueError:
            return datetime.datetime.utcnow().date()
    published_parsed = getattr(parsed, "updated_parsed", None) or getattr(parsed, "published_parsed", None)
    if published_parsed:
        return datetime.date(*published_parsed[:3])
    return datetime.datetime.utcnow().date()


def get_daily_sentiment(
    rss_urls: List[str] = None,
    custom_lexicon: Dict[str, float] = None,
) -> pd.DataFrame:
    """Scrape crypto news headlines, compute VADER sentiment, and aggregate by day.

    Args:
        rss_urls: Optional list of RSS feed URLs to scrape. If None, uses default crypto feeds.
        custom_lexicon: Optional dict of custom VADER tokens and sentiment scores.

    Returns:
        pd.DataFrame: daily average compound sentiment scores with Date as index.
            Columns: ['Average_Sentiment_Score']
    """
    rss_urls = rss_urls or DEFAULT_RSS_FEEDS
    analyzer = _load_sentiment_analyzer(custom_lexicon)
    headlines = _parse_headlines_from_feeds(rss_urls)

    if not headlines:
        return pd.DataFrame(columns=["Average_Sentiment_Score"]).astype({"Average_Sentiment_Score": float})

    rows = []
    for item in headlines:
        published_date = _normalize_date(item["published"])
        sentiment = analyzer.polarity_scores(item["title"])
        rows.append({"Date": published_date, "Average_Sentiment_Score": sentiment["compound"]})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    daily_avg = df.groupby("Date", as_index=True)["Average_Sentiment_Score"].mean()
    daily_df = daily_avg.to_frame()
    daily_df.index.name = "Date"
    return daily_df


if __name__ == "__main__":
    custom_terms = {
        "HODL": 2.0,
        "hodl": 2.0,
        "bullish": 2.5,
        "bearish": -2.5,
        "FOMO": 1.5,
    }
    sentiment_df = get_daily_sentiment(custom_lexicon=custom_terms)
    print(sentiment_df.head())
