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

from typing import Callable, Sequence

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    Orientationd,
    ToNumpyd,
    KeepLargestConnectedComponentd
)

from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored


class SegmentationCardiac(BasicInferTask):
    """
    This provides Inference Engine for pre-trained spleen segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        spatial_size=(96, 96, 96),
        target_spacing=(0.7, 0.7, 1.0),
        intensity=(-175, 250),
        sw_batch_size=2,
        overlap=0.25,
        description="A pre-trained model for volumetric (3D) segmentation of the spleen from CT image",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )
        self.model_state_dict = 'state_dict'
        self.spatial_size = [int(s) for s in spatial_size]
        self.target_spacing = target_spacing
        self.intensity = intensity
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.applied_labels = [0, 1]  # for post process KeepLargestConnectedComponentd

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadImaged(keys="image"),
            EnsureTyped(keys="image", device=data.get("device") if data else None),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys="image", pixdim=self.target_spacing, mode=("bilinear")),
            ScaleIntensityRanged(
                keys="image",
                a_min=self.intensity[0],
                a_max=self.intensity[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=self.spatial_size,
            sw_batch_size=self.sw_batch_size,
            overlap=self.overlap
        )

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(keys="pred", device=data.get("device") if data else None),
            AsDiscreted(keys="pred", argmax=True),
            Orientationd(keys=["pred"], axcodes="LPS"),
            KeepLargestConnectedComponentd(keys=["pred"], applied_labels=self.applied_labels),
            ToNumpyd(keys="pred"),
            Restored(keys="pred", ref_image="image"),
        ]
