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

import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from geniusrise_vision.base import VisionFineTuner


class ImageClassificationFineTuner(VisionFineTuner):
    def load_dataset(
        self,
        dataset_path: Union[str, None] = None,
        hf_dataset: Union[str, None] = None,
        is_multiclass: bool = False,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset for image classification from a local path or Hugging Face Datasets.

        Args:
            dataset_path (Union[str, None], optional): The local path to the dataset directory. Defaults to None.
            hf_dataset (Union[str, None], optional): The Hugging Face dataset identifier. Defaults to None.
            is_multiclass (bool, optional): Set to True for multi-class classification. Defaults to False.
            **kwargs: Additional arguments.

        Returns:
            Union[Dataset, DatasetDict, Optional[Dataset]]: The loaded dataset.
        """
        if dataset_path:
            return self._load_local_dataset(dataset_path, is_multiclass, **kwargs)
        elif hf_dataset:
            return load_dataset(hf_dataset, **kwargs)
        else:
            raise ValueError("Either 'dataset_path' or 'hf_dataset' must be provided")

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

        # Initialize a dictionary to hold image paths and their corresponding labels
        image_labels = defaultdict(list)

        # Aggregate labels for each image
        for label, index in label_to_index.items():
            class_dir = os.path.join(dataset_path, label)
            for img_file in os.listdir(class_dir):
                image_labels[img_file].append(index)

        # Load images and labels
        images, labels = [], []
        for img_file, label_indices in image_labels.items():
            try:
                img_path = os.path.join(dataset_path, self.model.config.id2label[label_indices[0]], img_file)
                with Image.open(img_path) as img:
                    # Process image using the Hugging Face processor
                    # processed_img = self.processor(images=img, return_tensors="pt")
                    images.append(pil_to_tensor(img))
                    labels.append(label_indices if is_multiclass else label_indices[0])
            except Exception as e:
                self.log.exception(f"Error loading image {os.path.basename(img_path)}: {e}")

        # Stack the images and labels into tensors
        images = torch.stack(images)  # type: ignore
        labels = torch.tensor(labels)  # type: ignore

        # Create a Hugging Face Dataset
        dataset = Dataset.from_dict({"image": images, "label": labels})

        return dataset

    def _create_label_index(self, dataset_path: str) -> Dict[str, int]:
        """
        Create a mapping from label names to indices.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Dict[str, int]: A dictionary mapping label names to indices.
        """
        classes = [d.name for d in os.scandir(dataset_path) if d.is_dir()]
        label_to_index = {cls_name: i for i, cls_name in enumerate(classes)}
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

    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        metric = evaluate.load("accuracy")
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
