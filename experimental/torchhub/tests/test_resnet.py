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

from pathlib import Path
from liga.types.vision import Image

import torchvision
import pandas as pd
from pyspark.sql import SparkSession

from liga.spark.functions import to_image
from liga.testing.utils import apply_model_spec


version = f"v{torchvision.__version__.split('+', maxsplit=1)[0]}"
work_dir = Path().absolute().parent.parent
image_path = f"{work_dir}/python/tests/assets/test_image.jpg"


def test_resnet_model_type():
    inputs_list = [pd.Series(Image(image_path))]
    results_list = apply_model_spec(
        {
            "name": "resnet50",
            "uri": f"torchhub:///pytorch/vision:{version}/resnet50",
        },
        inputs_list,
    )
    assert len(results_list) == 1
    series = results_list[0]
    assert len(series[0]) == 1000


def test_resnet(spark: SparkSession):
    spark.udf.register("to_image", to_image)
    for n in ["18", "34", "50", "101", "152"]:
        spark.sql(
            f"""
            CREATE MODEL resnet{n}
            OPTIONS (device="cpu", batch_size=32)
            USING "torchhub:///pytorch/vision:{version}/resnet{n}";
            """
        )
        result = spark.sql(
            f"""
        select ML_PREDICT(resnet{n}, to_image('{image_path}')) as pred
        """
        )
        assert len(result.head().pred) == 1000
