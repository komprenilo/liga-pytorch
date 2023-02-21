#  Copyright 2022 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
ResNet: Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
"""
import torchvision

from liga.pytorch.models.torch import (
    ClassificationModelType,
)

MODEL_TYPE = ClassificationModelType(
    name="resnet101", pretrained_fn=torchvision.models.resnet101
)
