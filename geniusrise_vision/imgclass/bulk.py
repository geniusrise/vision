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

from geniusrise import BatchInput, BatchOutput, State
from geniusrise_vision.base.bulk import ImageBulk
from torchvision import datasets, transforms
from datasets import Dataset, DatasetDict
from transformers import AutoModelForImageClassification, AutoProcessor
from collections import defaultdict
from typing import Dict, Optional, Union, List, Tuple
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
import json
import torch.nn as nn
import glob
from PIL import Image
import pandas as pd


class VisionClassificationBulk(ImageBulk):
    """
    VisionClassificationBulk class for bulk vision classification tasks using Hugging Face models.
    Inherits from VisionBulk.
    """
    
    def __init__(
        self, 
        input: BatchInput, 
        output: BatchOutput, 
        state: State, 
        **kwargs) -> None:

        super().__init__(input, output, state, **kwargs)

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
            return self._load_local_dataset(dataset_path, **kwargs)
        elif hf_dataset:
            return load_dataset(hf_dataset, **kwargs)
        else:
            raise ValueError("Either 'dataset_path' or 'hf_dataset' must be provided")
        


    def _load_local_dataset(self, dataset_path: str, image_size: Tuple[int, int] = (224, 224)) -> Dataset:
        """
        Load a dataset for image classification from a single folder containing images of various formats.

        Args:
            dataset_path (str): The path to the dataset directory.
            image_size (Tuple[int, int]): The size to resize images to.

        Returns:
            Dataset: The loaded dataset.
        """
        # Supported image formats
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif']

        # List of all image file paths in the dataset directory
        image_files = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in supported_formats]

        # Define a transform to resize the image and convert it to a tensor
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]) 

        # Create a dataset using lists for each field
        images = []
        paths = []
        for image_file in image_files:
            image_path = os.path.join(dataset_path, image_file)
            try:
                with Image.open(image_path).convert('RGB') as img:
                    img_tensor = transform(img)
                    # Debug: Check if img_tensor is a tensor
                    print(f"{image_file} is a tensor: {isinstance(img_tensor, torch.Tensor)}")
                    images.append(img_tensor)
                paths.append(image_path)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        # Convert lists to PyTorch Dataset
        return Dataset.from_dict({"image": images, "path": paths})

    def classify(
        self, 
        model_name, 
        model_class: str = "AutoModelForImageClassification",
        processor_class: str = "AutoProcessor",
        device_map: str | Dict | None = "auto",
        max_memory=None, 
        torchscript=False, 
        batch_size=32, 
        **kwargs) -> None:
        """
        Classifies vision data using a specified Hugging Face model.

        :param model_name: Name of the model to use.
        :param dataset_path: Path to the dataset.
        :param output_path: Path to save predictions.
        :param model_class: The class of the model (default: AutoModelForImageClassification).
        :param device_map: Device map for model parallelism.
        :param max_memory: Maximum memory for model parallelism.
        :param torchscript: Whether to use TorchScript.
        :param batch_size: Size of the batch for processing.
        :param kwargs: Additional keyword arguments.
        """
        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None
            processor_name = model_name
        
        # Load model
        self.model_name = model_name
        self.processor_name = processor_name
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.model_class = model_class
        self.processor_class = processor_class
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.batch_size = batch_size

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        self.model, self.processor = self.load_models(
            model_name=self.model_name,
            processor_name=self.processor_name,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            model_class=self.model_class,
            processor_class=self.processor_class,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            **self.model_args,
        )

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset = self.load_dataset(self.input.input_folder)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Process and classify in batches
        for batch_idx in range(0, len(dataset), batch_size):
            batch = dataset[batch_idx:batch_idx + batch_size]
            image_tensors = batch['image']

            # Convert each list of pixel values to a tensor
            tensor_batch = [torch.tensor(image).float() for image in image_tensors]

            # Stack the tensors
            inputs = torch.stack(tensor_batch).to(device)
                    
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            self._save_predictions(predictions, batch, self.output.output_folder, batch_idx)

    def _save_predictions(
        self, 
        predictions: torch.Tensor,
        batch: List[str], 
        output_path: str, 
        batch_index) -> None:
        # Convert predictions to labels
        labels = [self.model.config.id2label[prediction.item()] for prediction in predictions]

        # Create a list of dictionaries with image paths and their corresponding predictions
        results = [{"image_path": image_path, "prediction": label} for image_path, label in zip(batch, labels)]

        # Save to JSON
        file_name = os.path.join(output_path, f"predictions_batch_{batch_index}.json")
        with open(file_name, 'w') as file:
            json.dump(results, file, indent=4)