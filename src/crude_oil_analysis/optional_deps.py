"""Optional third-party dependencies with graceful fallbacks.

Importing this module never has side effects: it does not print and never
shells out to ``pip``. If an optional package is missing, an availability flag
is set to ``False`` and a minimal dummy stand-in is provided so the rest of the
program keeps working (with the corresponding feature disabled).

Install the optional extras with ``pip install xgboost fpdf2 feedparser`` (all
are included in ``requirements.txt``).
"""
from __future__ import annotations

# --- XGBoost (modelling; falls back to RandomForest in the forecaster) ------
try:
    import xgboost as xgb  # type: ignore
    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when xgboost is absent
    xgb = None  # type: ignore
    XGB_AVAILABLE = False


# --- feedparser (news collection) -------------------------------------------
try:
    import feedparser  # type: ignore
    FEEDPARSER_AVAILABLE = True
except ImportError:  # pragma: no cover
    FEEDPARSER_AVAILABLE = False

    class _DummyFeedparser:
        @staticmethod
        def parse(*args, **kwargs):
            return {"entries": []}

    feedparser = _DummyFeedparser()  # type: ignore


# --- fpdf2 (PDF reporting) --------------------------------------------------
try:
    from fpdf import FPDF  # type: ignore
    from fpdf.enums import XPos, YPos  # type: ignore
    PDF_AVAILABLE = True
    FPDF_CLASS = FPDF
except ImportError:  # pragma: no cover
    PDF_AVAILABLE = False
    XPos = None  # type: ignore
    YPos = None  # type: ignore

    class _DummyFPDF:
        """No-op stand-in so report generation degrades gracefully."""

        def add_page(self, *a, **k): pass
        def set_font(self, *a, **k): pass
        def set_margins(self, *a, **k): pass
        def add_font(self, *a, **k): pass
        def cell(self, *a, **k): pass
        def multi_cell(self, *a, **k): pass
        def ln(self, *a, **k): pass
        def image(self, *a, **k): pass
        def get_y(self): return 0
        def get_string_width(self, *a, **k): return 0
        def output(self, *a, **k): pass

    FPDF_CLASS = _DummyFPDF


__all__ = [
    "xgb", "XGB_AVAILABLE",
    "feedparser", "FEEDPARSER_AVAILABLE",
    "FPDF_CLASS", "PDF_AVAILABLE", "XPos", "YPos",
]
