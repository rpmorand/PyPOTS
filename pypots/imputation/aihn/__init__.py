"""
The package of the partially-observed time-series imputation and forecasting model AIHN.

"""

# Created by Knut Streoemmen <knut.stroemmen@unibe.com> and Rafael Morand <rafael.morand@unibe.ch>
# License: BSD-3-Clause

from .model import AIHN
from .data import DatasetMainFrequencyBased

__all__ = [
    "AIHN",
    "DatasetMainFrequencyBased",
]
