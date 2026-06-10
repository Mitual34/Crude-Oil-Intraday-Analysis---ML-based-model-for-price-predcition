"""Smoke tests for the CLI argument parser."""
import pytest

from crude_oil_analysis.cli import build_parser


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.interval == "30m"
    assert args.days == 30
    assert args.email is None
    assert args.gcs_bucket is None


def test_parser_rejects_invalid_interval():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--interval", "1d"])


def test_parser_accepts_options():
    args = build_parser().parse_args(
        ["--interval", "15m", "--days", "10", "--email", "a@b.com", "--gcs-bucket", "my-bucket"]
    )
    assert args.interval == "15m"
    assert args.days == 10
    assert args.email == "a@b.com"
    assert args.gcs_bucket == "my-bucket"
