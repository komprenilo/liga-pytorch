#  Copyright 2021 Rikai Authors
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

import warnings
from pathlib import Path
from urllib.parse import urlparse

warnings.filterwarnings("ignore", category=DeprecationWarning)  # noqa

# Third Party
import pytest
import torch
import torchvision
from pyspark.sql import Row, SparkSession

from liga.mlflow import CONF_MLFLOW_TRACKING_URI
from liga.logging import logger
from ligavision.spark import init_session
from ligavision.spark.types import Image


@pytest.fixture(scope="session")
def two_flickr_images() -> list:
    return [
        Image.read(uri)
        for uri in [
            "http://farm2.staticflickr.com/1129/4726871278_4dd241a03a_z.jpg",
            "http://farm4.staticflickr.com/3726/9457732891_87c6512b62_z.jpg",
        ]
    ]


@pytest.fixture(scope="session")
def two_flickr_rows(two_flickr_images: list) -> list:
    return [Row(image=image) for image in two_flickr_images]


@pytest.fixture(scope="session")
def tracking_uri(tmp_path_factory) -> str:
    tmp_path = tmp_path_factory.mktemp("mlflow")
    tmp_path.mkdir(parents=True, exist_ok=True)
    tracking_uri = "sqlite:///" + str(tmp_path / "tracking.db")
    return tracking_uri


@pytest.fixture(scope="session")
def spark(tracking_uri, tmp_path_factory) -> SparkSession:
    logger.info(f"mlflow tracking uri for spark: {tracking_uri}")
    warehouse_path = tmp_path_factory.mktemp("warehouse")
    spark = init_session(conf=dict(
        [
            ("spark.port.maxRetries", 128),
            ("spark.sql.warehouse.dir", str(warehouse_path)),
            (
                "spark.rikai.sql.ml.registry.test.impl",
                "net.xmacs.liga.model.testing.TestRegistry",
            ),
            (
                "spark.rikai.sql.ml.catalog.impl",
                "net.xmacs.liga.model.SimpleCatalog",
            ),
            (
                CONF_MLFLOW_TRACKING_URI,
                tracking_uri,
            ),
        ]
    ))
    return spark

@pytest.fixture
def asset_path() -> Path:
    return Path(__file__).parent / "assets"
