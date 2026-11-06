"""Packaging alias; model construction lives in unchanged model_overnight.py.

Importing this alias does not construct a model. The historical model source is
the separately retained model_overnight.py, not this newly named wrapper.
"""
from model_overnight import (APPROVED_BACKBONES, BASE_BACKBONE_NAME,
    TINY_BACKBONE_NAME, NUM_CLASSES, build_overnight_model, validate_model_config)

__all__ = ["APPROVED_BACKBONES", "BASE_BACKBONE_NAME", "TINY_BACKBONE_NAME",
           "NUM_CLASSES", "build_overnight_model", "validate_model_config"]
