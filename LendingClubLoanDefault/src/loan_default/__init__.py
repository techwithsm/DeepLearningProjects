"""Lending Club loan default prediction with Keras / TensorFlow."""
import os

# Silence TensorFlow's C++ start-up chatter before any submodule imports it.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

__version__ = "1.0.0"
