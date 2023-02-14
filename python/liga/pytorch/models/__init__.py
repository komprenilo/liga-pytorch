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

"""Rikai-implemented PyTorch models and executors."""

import importlib

torchvision_found = importlib.util.find_spec("torchvision") is not None

if torchvision_found:
    import liga.pytorch.models.convnext
    import liga.pytorch.models.efficientnet
    import liga.pytorch.models.fasterrcnn
    import liga.pytorch.models.feature_extractor
    import liga.pytorch.models.keypointrcnn
    import liga.pytorch.models.maskrcnn
    import liga.pytorch.models.resnet
    import liga.pytorch.models.retinanet
    import liga.pytorch.models.ssd
    import liga.pytorch.models.ssd_class_scores
