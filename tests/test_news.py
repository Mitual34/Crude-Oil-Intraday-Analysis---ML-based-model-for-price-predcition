"""Tests for sentiment scoring (no network)."""
from crude_oil_analysis.news import analyze_sentiment


def test_empty_headlines_returns_zero():
    assert analyze_sentiment([]) == 0.0


def test_positive_headline_scores_positive():
    assert analyze_sentiment(["Oil prices surge on excellent demand and great outlook"]) > 0


def test_negative_headline_scores_negative():
    assert analyze_sentiment(["Terrible crash: oil plunges on awful, horrible recession fears"]) < 0


def test_result_is_rounded_to_two_dp():
    score = analyze_sentiment(["A wonderful and fantastic rally"])
    assert score == round(score, 2)
