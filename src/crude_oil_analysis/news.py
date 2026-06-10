"""News collection (Google News RSS) and headline sentiment scoring."""
from __future__ import annotations

import urllib.parse

from textblob import TextBlob

from .optional_deps import FEEDPARSER_AVAILABLE, feedparser


class GoogleNews:
    """Tiny wrapper around the Google News RSS feed."""

    def __init__(self, lang: str = "en", country: str = "US"):
        self.lang = lang.lower()
        self.country = country.upper()
        self.BASE_URL = "https://news.google.com/rss"

    def search(self, query, from_=None, to_=None, exclude=None):
        # Only attempt news collection if feedparser is available
        if not FEEDPARSER_AVAILABLE:
            return {"entries": []}

        params = {
            "q": query,
            "hl": f"{self.lang}-{self.country}",
            "gl": self.country,
            "ceid": f"{self.country}:{self.lang}",
        }

        if from_ or to_:
            date_range = []
            if from_:
                date_range.append(f'after:{from_.replace("-", "/")}')
            if to_:
                date_range.append(f'before:{to_.replace("-", "/")}')
            params["q"] += " " + " ".join(date_range)

        if exclude:
            if isinstance(exclude, list):
                exclude_str = " ".join([f"-{term}" for term in exclude])
            else:
                exclude_str = f"-{exclude}"
            params["q"] += " " + exclude_str

        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        feed = feedparser.parse(url)
        return feed


def analyze_sentiment(headlines) -> float:
    """Return the mean TextBlob polarity of ``headlines`` rounded to 2 dp."""
    if not headlines:
        return 0.0

    sentiments = []
    for title in headlines:
        try:
            analysis = TextBlob(title)
            sentiments.append(analysis.sentiment.polarity)
        except Exception:
            continue

    if sentiments:
        return round(sum(sentiments) / len(sentiments), 2)
    return 0.0
