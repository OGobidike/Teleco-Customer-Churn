"""Telecom customer churn: data, a leak-proof pipeline, evaluation, an audit and an API."""

import os

# Keep TensorFlow's start-up chatter out of the console (it changes no numbers).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__version__ = "2.0.0"
