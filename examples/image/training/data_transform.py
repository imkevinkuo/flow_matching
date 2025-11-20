# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, Resize, ToDtype, ToImage


def get_train_transform(image_size=None):
    transform_list = [
        ToImage(),
    ]
    if image_size is not None:
        transform_list.append(Resize((image_size, image_size), antialias=True))
    transform_list.extend([
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ])
    return Compose(transform_list)
