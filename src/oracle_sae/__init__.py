"""Criterion-driven feature discovery and cross-model activation matching."""

from oracle_sae.pipeline import inspect_model, match_reports
from oracle_sae.schema import FeatureCard, FeatureFingerprint, InspectionReport

__all__ = [
    "FeatureCard",
    "FeatureFingerprint",
    "InspectionReport",
    "inspect_model",
    "match_reports",
]

__version__ = "0.2.0"
