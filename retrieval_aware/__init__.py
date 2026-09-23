"""Retrieval-Aware AutoGEO (E7) experiment package.

The package is intentionally isolated from the original AutoGEO pipeline.
The formal retrieval environment is a split-isolated pooled local corpus.
Importing the package never performs network, model, rewrite, or evaluation
work.
"""

from .config import E7Config
from .protocol import ExperimentTrack, ProtocolId, ProtocolSpec

__all__ = ["E7Config", "ExperimentTrack", "ProtocolId", "ProtocolSpec"]
__version__ = "0.6.0-dev-target-selection"
