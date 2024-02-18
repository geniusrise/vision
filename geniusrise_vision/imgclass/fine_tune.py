# 🧠 Geniusrise
# Copyright (C) 2023  geniusrise.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from typing import Dict, Optional, Union
from torch.utils.data import TensorDataset
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from transformers import EvalPrediction
from sklearn.metrics import accuracy_score
from torchvision.transforms.functional import pil_to_tensor

from geniusrise_vision.base import VisionFineTuner


class ImageClassificationFineTuner(VisionFineTuner):
    def load_dataset(
        self,
        dataset_path: str,
        is_multiclass: bool = False,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset for image classification from a local path or Hugging Face Datasets.

        Args:
            dataset_path (Union[str, None], optional): The local path to the dataset directory. Defaults to None.
            is_multiclass (bool, optional): Set to True for multi-class classification. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            Union[Dataset, DatasetDict, Optional[Dataset]]: The loaded dataset.
        """

        if self.use_huggingface_dataset:
            dataset = load_dataset(self.huggingface_dataset)
        elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
            dataset = load_from_disk(dataset_path, **kwargs)
        else:
            dataset = self._load_local_dataset(dataset_path, is_multiclass, **kwargs)

        if hasattr(self, "map_data") and self.map_data:
            fn = eval(self.map_data)  # type: ignore
            dataset = dataset.map(fn)
        else:
            dataset = dataset

        return dataset

    def _load_local_dataset(
        self, dataset_path: str, is_multiclass: bool = False, **kwargs
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset for image classification.

        Args:
            dataset_path (str): The path to the dataset directory.
            is_multiclass (bool, optional): Set to True for multi-class classification. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            Union[Dataset, DatasetDict, Optional[Dataset]]: The loaded dataset.
        """
        label_to_index = self._create_label_index(dataset_path)
        self._save_label_index(label_to_index)

        image_labels = defaultdict(list)  # type: ignore
        num_classes = len(label_to_index)

        # Determine if the dataset has 'train' and 'test' directories
        has_train_test_split = all(os.path.isdir(os.path.join(dataset_path, split)) for split in ["train", "test"])
        splits = ["train", "test"] if has_train_test_split else [""]

        # Iterate through splits and class directories
        for split in splits:
            split_dir = os.path.join(dataset_path, split) if split else dataset_path
            for label in label_to_index:
                class_dir = os.path.join(split_dir, label)
                if os.path.isdir(class_dir):
                    for img_file in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_file)
                        if os.path.isfile(img_path):
                            label_idx = label_to_index[label]

                            # Use image file name for label aggregation
                            if img_file in image_labels and is_multiclass:
                                # Add the new label index if it's not already present
                                if label_idx not in image_labels[img_file]["labels"]:
                                    image_labels[img_file]["labels"].append(label_idx)
                            else:
                                # Store both the full path and label
                                image_labels[img_file] = {"path": img_path, "labels": [label_idx]}

        images, labels = [], []
        for img_file, data in image_labels.items():
            img_path = data["path"]  # Retrieve the full path
            label_indices = data["labels"]
            try:
                with Image.open(img_path).convert("RGB") as img:
                    # Convert image to tensor and then to float
                    image_tensor = pil_to_tensor(img).float()
                    if image_tensor.nelement() == 0:  # Check for empty tensor
                        continue
                    images.append(image_tensor)

                    if is_multiclass:
                        # OneHot Encoding
                        label_tensor = torch.zeros(num_classes, dtype=torch.long)
                        label_tensor.scatter_(0, torch.tensor(label_indices, dtype=torch.long), 1)
                    else:
                        label_tensor = torch.tensor(label_indices, dtype=torch.long)

                    labels.append(label_tensor)
            except Exception as e:
                self.log.exception(f"Error loading image {os.path.basename(img_path)}: {e}")

        if not images:
            raise RuntimeError("No images were loaded.")

        tensor_dataset = TensorDataset(torch.stack(images), torch.stack(labels))
        return tensor_dataset

    def custom_vision_collator(self, batch):
        """
        Collates batches of our dataset.

        Args:
            batch: A batch from the dataset.

        Returns:
            A dictionary with the keys 'pixel_values' and 'labels'.
        """
        # Assuming each item in the batch is a tuple (image, label)
        pixel_values = [item[0] for item in batch]  # Extract all image tensors
        labels = [item[1] for item in batch]  # Extract all label tensors

        # Stack the extracted image tensors and label tensors
        pixel_values = torch.stack(pixel_values)
        labels = torch.stack(labels)

        return {"pixel_values": pixel_values, "labels": labels}

    def _create_label_index(self, dataset_path: str) -> Dict[str, int]:
        """
        Create a mapping from label names to indices.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dict[str, int]: A dictionary mapping label names to indices.
        """
        # Check if 'train' and 'test' directories exist
        has_train_test_split = all(os.path.isdir(os.path.join(dataset_path, split)) for split in ["train", "test"])

        # Initialize an empty set for class names
        class_names = set()  # type: ignore

        # Choose the directories to scan for class names based on the dataset structure
        if has_train_test_split:
            for split in ["train", "test"]:
                class_dir = os.path.join(dataset_path, split)
                class_names.update(d.name for d in os.scandir(class_dir) if d.is_dir())
        else:
            class_names.update(d.name for d in os.scandir(dataset_path) if d.is_dir())

        label_to_index = {cls_name: i for i, cls_name in enumerate(sorted(class_names))}
        return label_to_index

    def _save_label_index(self, label_to_index: Dict[str, int]):
        """
        Save the label index in the model's configuration.

        Args:
            label_to_index (Dict[str, int]): A dictionary mapping label names to indices.
        """
        if self.model.config:
            self.model.config.label2id = label_to_index
            self.model.config.id2label = {id: label for label, id in label_to_index.items()}
        else:
            self.log.warning("Model configuration is not loaded. Cannot save label index.")

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
        }
