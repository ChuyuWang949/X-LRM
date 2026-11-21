# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted for X-LRM for CBCT reconstruction.

from abc import ABC, abstractmethod
import json
import numpy as np
import torch
from megfile import smart_open, smart_path_join, smart_exists
from dataclasses import dataclass

@dataclass
class DatasetCfgCommon:
    eval:  bool = False
    shuffle: bool = True

class BaseDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root_dirs: list[str], meta_path: str):
        super().__init__()
        self.root_dirs = root_dirs
        self.ids = self._load_ids(meta_path)

    def __len__(self):
        return len(self.ids)

    @abstractmethod
    def inner_get_item(self, idx):
        pass

    def __getitem__(self, idx):
        try:
            return self.inner_get_item(idx)
        except Exception as e:
            print(f"[DEBUG-DATASET] Error when loading {self.ids[idx]}")
            # return self.__getitem__(idx+1)
            raise e

    @staticmethod
    def _load_ids(meta_path: str):
        # meta_path is a json file
        with open(meta_path, 'r') as f:
            ids = json.load(f)
        return ids

    @staticmethod
    def _locate_datadir(root_dir, id, locator: str):
        datadir = smart_path_join(root_dir, id, locator)
        if smart_exists(datadir):
            return root_dir
        raise FileNotFoundError(f"Cannot find valid data directory for id {id}")
