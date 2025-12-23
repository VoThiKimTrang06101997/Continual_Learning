# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmdet.structures import SampleList

