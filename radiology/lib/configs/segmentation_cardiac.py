# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, Optional, Union

import lib.infers
import lib.trainers
from lib.networks.network import network
from lib.networks.utils import download_ckp

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum
from monailabel.utils.others.generic import download_file, strtobool


logger = logging.getLogger(__name__)


class SegmentationCardiac(TaskConfig):
    def __init__(self):
        super().__init__()

        self.epistemic_enabled = None
        self.epistemic_samples = None

    def init(self, name: str, model_dir: str, conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # Labels
        self.labels = {
            "cardiac": 1,
        }

        # Network name
        self.network_name = self.conf.get("network", "unetcnx_x0")

        # Model Files
        self.path = [
            os.path.join(self.model_dir, self.network_name, f"best_model.pth"),  # pretrained
            os.path.join(self.model_dir, self.network_name, f"out_model.pth"),  # published
        ]

        # download checkpoint
        download_ckp(self.path[0], self.conf.get('download_ckp_id', None))

        # Transform config
        self.target_spacing = [0.7, 0.7, 1.0]  # target space for image
        self.spatial_size = [96, 96, 96]  # train input size
        self.intensity = [-175, 250]

        # Network
        self.network = network(
            self.network_name,
            in_channels=1,
            out_channels=len(self.labels) + 1
        )

        # Infer config
        self.sw_batch_size = 2
        self.overlap = 0.25

        # Others
        self.epistemic_enabled = strtobool(conf.get("epistemic_enabled", "false"))
        self.epistemic_samples = int(conf.get("epistemic_samples", "5"))
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled}; Samples: {self.epistemic_samples}")


    def infer(self) -> Union[InferTask, Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationCardiac(
            path=self.path,
            network=self.network,
            labels=self.labels,
            spatial_size=self.spatial_size,
            target_spacing=self.target_spacing,
            intensity=self.intensity,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap,
            preload=strtobool(self.conf.get("preload", "false")),
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        output_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        task: TrainTask = lib.trainers.SegmentationSpleen(
            model_dir=output_dir,
            network=self.network,
            description="Train Spleen Segmentation Model",
            load_path=load_path,
            publish_path=self.path[1],
            labels=self.labels,
            disable_meta_tracking=False,
        )
        return task

    def strategy(self) -> Union[None, Strategy, Dict[str, Strategy]]:
        strategies: Dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> Union[None, ScoringMethod, Dict[str, ScoringMethod]]:
        methods: Dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum(),
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = EpistemicScoring(
                model=self.path,
                network=self.network,
                transforms=lib.infers.SegmentationCardiac(None).pre_transforms(),
                num_samples=self.epistemic_samples,
            )
        return methods
