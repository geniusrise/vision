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
import torch
import json
import torch.nn as nn
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_vision.base import VisionFineTuner
from typing import Union, Dict, Optional, Callable
from PIL import Image
import albumentations as A
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, DatasetDict
from transformers import MaskFormerImageProcessor
from transformers import EvalPrediction, Trainer, TrainingArguments


class SegmentationFineTuner(VisionFineTuner):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        super().__init__(input=input, output=output, state=state)

    def load_dataset(
        self,
        dataset_path: Union[str, None] = None,
        hf_dataset: Union[str, None] = None,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset for ocr from a local path or Hugging Face Datasets.

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

    def _load_local_dataset(self, dataset_path: str, **kwargs) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        dataset = []

        # Check if there are any subdirectories in dataset_path
        if any(os.path.isdir(os.path.join(dataset_path, item)) for item in os.listdir(dataset_path)):
            # If there are subdirectories, assume they are categories
            for category in os.listdir(dataset_path):
                category_path = os.path.join(dataset_path, category)
                if not os.path.isdir(category_path):
                    continue

                dataset += self._process_directory(category_path)
        else:
            # If there are no subdirectories, process the dataset_path directly
            dataset = self._process_directory(dataset_path)

        return dataset

    def _process_directory(self, directory_path: str) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        processed_data = []
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".jpg"):
                image_name = os.path.splitext(file_name)[0]

                # Load Image
                image_path = os.path.join(directory_path, file_name)
                image = Image.open(image_path).convert("RGB")

                # Load Segmentation Bitmap
                seg_path = os.path.join(directory_path, f"{image_name}_seg.png")
                seg_image = Image.open(seg_path).convert("L") if os.path.exists(seg_path) else None

                # Load JSON Metadata
                json_path = os.path.join(directory_path, f"{image_name}.json")
                metadata = {}
                inst2class = {}
                if os.path.exists(json_path):
                    with open(json_path, "r") as json_file:
                        metadata = json.load(json_file)
                        # Extract inst2class mapping from metadata (assuming it's in the format of panoptic segmentation datasets)
                        inst2class = {
                            segment["id"]: segment["category_id"] for segment in metadata.get("segments_info", [])
                        }

                # Add to dataset
                dataset_entry = {
                    "pixel_values": image,
                    "label": seg_image,
                    "metadata": metadata,
                    "inst2class": inst2class,
                }
                processed_data.append(dataset_entry)

        return processed_data

    def preprocess_data(self, **kwargs):
        try:
            self.input.copy_from_remote()
            train_dataset_path = os.path.join(self.input.get(), "train")
            eval_dataset_path = os.path.join(self.input.get(), "test")

            # # Create a preprocessor
            # preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

            # Load datasets
            raw_train_dataset = self.load_dataset(train_dataset_path, **kwargs)
            self.train_dataset = self.apply_transforms(raw_train_dataset, self.train_transforms)
            # self.train_dataloader = DataLoader(self.train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

            if self.evaluate:
                raw_eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
                self.eval_dataset = self.apply_transforms(raw_eval_dataset, self.eval_transforms)
                # self.eval_dataloader = DataLoader(self.eval_dataset, batch_size=2, shuffle=False, collate_fn=self.collate_fn)

            print("Inspecting Train Dataset:")
            self.inspect_dataset(self.train_dataset)

        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise e

    def inspect_dataset(self, dataset, num_samples=5):
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break

            print(f"Sample {i}:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor with shape {value.shape}")
                elif isinstance(value, list):
                    if value and isinstance(value[0], torch.Tensor):
                        print(f"  {key}: List of Tensors with shapes {[v.shape for v in value]}")
                    else:
                        print(f"  {key}: List with length {len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
            print()

    # def collate_fn(self, batch):
    #     inputs = list(zip(*batch))
    #     images = inputs[0]
    #     segmentation_maps = inputs[1]
    #     # this function pads the inputs to the same size,
    #     # and creates a pixel mask
    #     # actually padding isn't required here since we are cropping
    #     batch = preprocessor(
    #         images,
    #         segmentation_maps=segmentation_maps,
    #         return_tensors="pt",
    #     )

    # return batch

    def apply_transforms(self, dataset, transform_function):
        """
        Apply transformations to each entry in the dataset.

        Args:
            dataset: The dataset to be transformed.
            transform_function: The transformation function to apply.

        Returns:
            Transformed dataset.
        """
        transformed_dataset = []
        for entry in dataset:
            transformed_entry = transform_function(entry)
            transformed_dataset.append(transformed_entry)
        return transformed_dataset

    def remap_labels_to_continuous(self, mask):
        unique_values = np.unique(mask)
        mapping = {old_label: new_label for new_label, old_label in enumerate(unique_values)}
        remapped_mask = np.copy(mask)
        # Apply the mapping
        for old_label, new_label in mapping.items():
            remapped_mask[mask == old_label] = new_label
        return remapped_mask, mapping

    def train_transforms(self, example_batch):
        # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
        # Apply jitter directly to the PIL image
        # image = jitter(example_batch['pixel_values'])
        # Directly use the single label
        image = example_batch["pixel_values"]
        label = example_batch["label"]

        if self.model.config.__class__.__name__ in ["MaskFormerConfig", "Mask2FormerConfig"]:
            image = np.array(image)
            mask = np.array(label)

            # Define Albumentations transform
            ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
            ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=1333),
                    A.RandomCrop(width=512, height=512),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
                ]
            )

            # Apply Albumentations transformations
            transformed = transform(image=image, mask=mask)
            image, segmentation_map = transformed["image"], transformed["mask"]

            # convert to C, H, W
            image = image.transpose(2, 0, 1)

            # Create a preprocessor
            preprocessor = MaskFormerImageProcessor(
                ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False
            )

            inputs = preprocessor([image], segmentation_maps=[segmentation_map], return_tensors="pt")
            # inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k,v in inputs.items()}

            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                else:
                    print(f"{key}: {value}")
                    print(key, [x.shape for x in value])

            # return image, segmentation_map

        elif self.model.config.__class__.__name__ in ["OneFormerConfig"]:
            inputs = self.processor([image], [label], task_inputs=["semantic"], return_tensors="pt")
        elif self.model.config.__class__.__name__ in {
            "MobileNetV2Config",
            "DPTConfig",
            "MobileViTConfig",
            "MobileViTV2Config",
        }:
            image = np.array(image)
            mask = np.array(label)
            unique_values = np.unique(mask)
            print("Mask is:", mask)
            print("Unique values in mask:", unique_values)

            # Apply the remapping function
            remapped_mask, mapping = self.remap_labels_to_continuous(mask)
            print("Remapped Mask:\n", remapped_mask)
            print("Mapping:\n", mapping)

            # Define Albumentations transform
            ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
            ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

            transform = A.Compose(
                [
                    A.LongestMaxSize(max_size=1333),
                    A.RandomCrop(width=512, height=512),
                    # A.HorizontalFlip(p=0.5),
                    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
                ]
            )

            # Apply Albumentations transformations
            transformed = transform(image=image, mask=remapped_mask)
            image, label = transformed["image"], transformed["mask"]

            # # convert to C, H, W
            # image = image.transpose(2,0,1)

            # Process the image for classification/semantic segmentation models without inst2class
            inputs = self.processor(
                image, reduce_labels=True, do_resize=False, do_rescale=False, do_normalize=False, return_tensors="pt"
            )
            # Add label to inputs if it's necessary for your task
            inputs["labels"] = torch.tensor(label, dtype=torch.long)

        elif self.model.config.__class__.__name__ in {"Data2VecVisionConfig"}:
            # Process the image for classification/semantic segmentation models without inst2class
            inputs = self.processor(image, label, reduce_labels=True, return_tensors="pt")
        else:
            # Process the image and label
            inputs = self.processor([image], [label], return_tensors="pt")
            # Print each key and its corresponding shape in inputs
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                else:
                    print(f"{key}: {value}")
                    print(key, [x.shape for x in value])

        return inputs

    def eval_transforms(self, example_batch):
        # Directly use the single image
        image = example_batch["pixel_values"]
        # Directly use the single label
        label = example_batch["label"]
        # Process the image and label
        inputs = self.processor([image], [label], return_tensors="pt")
        return inputs

    def custom_vision_collator(self, batch):
        """
        Collates batches of our dataset. The batch is a list of dictionaries
        where each dictionary contains 'pixel_values' and 'label' as keys.

        Args:
            batch: A batch from the dataset.

        Returns:
            The collated batch.
        """

        if self.model.config.__class__.__name__ in ["MaskFormerConfig", "Mask2FormerConfig", "OneFormerConfig"]:
            pixel_values = [item["pixel_values"] for item in batch]
            # pixel_values = [torch.tensor(image, dtype=torch.float32) if not isinstance(image, torch.Tensor) else image for image in pixel_values]

            # Check shapes of pixel_values before and after stacking
            print("Shapes of pixel_values before stacking:", [pv.shape for pv in pixel_values])
            pixel_values = torch.stack(pixel_values).squeeze(1)
            print("Shape of pixel_values after stacking and squeezing:", pixel_values.shape)

            pixel_mask = [item["pixel_mask"] for item in batch]
            mask_labels = [item["mask_labels"] for item in batch]
            class_labels = [item["class_labels"] for item in batch]

            print("Shapes of pixel_mask:", [pm.shape for pm in pixel_mask])
            pixel_mask = torch.stack(pixel_mask).squeeze(1)
            print("Shape of pixel_values after stacking and squeezing:", pixel_mask.shape)
            # Check shapes of pixel_mask, mask_labels, and class_labels
            print("Shapes of pixel_mask:", [pm.shape for pm in pixel_mask])
            print("Shapes of each item in mask_labels:", [[ml.shape for ml in sublist] for sublist in mask_labels])
            print("Shapes of each item in class_labels:", [[cl.shape for cl in sublist] for sublist in class_labels])

            return {
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
                "class_labels": class_labels,
                "mask_labels": mask_labels,
            }

            # inputs = list(zip(*batch))
            # images = inputs[0]
            # segmentation_maps = inputs[1]
            # # this function pads the inputs to the same size,
            # # and creates a pixel mask
            # # actually padding isn't required here since we are cropping
            # preprocessor = MaskFormerImageProcessor(ignore_index=0, reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
            # batch = preprocessor(
            #     images,
            #     segmentation_maps=segmentation_maps,
            #     return_tensors="pt",
            # )

            # return batch

        elif self.model.config.__class__.__name__ in ["MobileNetV2Config", "DPTConfig", "SegformerConfig"]:
            pixel_values = [item["pixel_values"] for item in batch]
            pixel_values = [
                torch.tensor(image, dtype=torch.float32) if not isinstance(image, torch.Tensor) else image
                for image in pixel_values
            ]
            labels = [item["labels"] for item in batch]
            pixel_values = torch.stack(pixel_values).squeeze(1)
            labels = torch.stack(labels).squeeze(1).long()
            return {"pixel_values": pixel_values, "labels": labels}

        else:
            pixel_values = [item["pixel_values"] for item in batch]
            pixel_values = [
                torch.tensor(image, dtype=torch.float32) if not isinstance(image, torch.Tensor) else image
                for image in pixel_values
            ]
            labels = [item["labels"] for item in batch]
            pixel_values = torch.stack(pixel_values).squeeze(1)
            labels = torch.stack(labels).squeeze(1).long()
            return {"pixel_values": pixel_values, "labels": labels}

    def _create_label_index(self, id2label: Dict[int, str]) -> Dict[str, int]:
        """
        Adjust the method to directly use the id2label mapping for segmentation.

        Args:
            id2label (Dict[int, str]): A dictionary mapping ids to label names.

        Returns:
            Dict[str, int]: A dictionary mapping label names to indices.
        """
        label_to_index = {label: id for id, label in id2label.items()}
        return label_to_index

    def _save_label_index(self, label_to_index: Dict[str, int]):
        """
        Save the label index for segmentation in the model's configuration.

        Args:
            label_to_index (Dict[str, int]): A dictionary mapping label names to indices.
        """
        if self.model.config:
            self.model.config.label2id = label_to_index
            self.model.config.id2label = {id: label for label, id in label_to_index.items()}
        else:
            self.log.warning("Model configuration is not loaded. Cannot save label index for segmentation.")

    def fine_tune(  # type: ignore
        self,
        model_name: str,
        processor_name: str,
        num_train_epochs: int,
        per_device_batch_size: int,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        device_map: str | dict = "auto",
        evaluate: bool = False,
        subtask: str = "semantic",
        map_data: Optional[Callable] = None,
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = True,
        hf_create_pr: bool = False,
        **kwargs,
    ):
        """
        Fine-tunes a pre-trained Hugging Face model.

        Args:
            model_name (str): The name of the pre-trained model.
            processor_name (str): The name of the pre-trained processor.
            num_train_epochs (int): The total number of training epochs to perform.
            per_device_batch_size (int): The batch size per device during training.
            model_class (str, optional): The model class to use. Defaults to "AutoModel".
            processor_class (str, optional): The processor class to use. Defaults to "AutoProcessor".
            device_map (str | dict, optional): The device map for distributed training. Defaults to "auto".
            evaluate (bool, optional): Whether to evaluate the model after training. Defaults to False.
            map_data (Callable, optional): A function to map data before training. Defaults to None.
            hf_repo_id (str, optional): The Hugging Face repo ID. Defaults to None.
            hf_commit_message (str, optional): The Hugging Face commit message. Defaults to None.
            hf_token (str, optional): The Hugging Face token. Defaults to None.
            hf_private (bool, optional): Whether to make the repo private. Defaults to True.
            hf_create_pr (bool, optional): Whether to create a pull request. Defaults to False.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            None
        """
        try:
            # Save everything
            self.model_name = model_name
            self.processor_name = processor_name
            self.num_train_epochs = num_train_epochs
            self.per_device_batch_size = per_device_batch_size
            self.model_class = model_class
            self.processor_class = processor_class
            self.device_map = device_map
            self.evaluate = evaluate
            self.subtask = subtask
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr
            self.map_data = map_data

            model_kwargs = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}

            self.load_models(
                model_name=self.model_name,
                processor_name=self.processor_name,
                model_class=self.model_class,
                processor_class=self.processor_class,
                device_map=self.device_map,
                **model_kwargs,
            )

            # Load dataset
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {k.replace("trainer_", ""): v for k, v in kwargs.items() if "trainer_" in k}
            training_kwargs = {k.replace("training_", ""): v for k, v in kwargs.items() if "training_" in k}

            print(len(self.model.config.id2label))

            # Create training arguments
            training_args = TrainingArguments(
                output_dir=os.path.join(self.output.output_folder, "model"),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                **training_kwargs,
            )

            # Create trainer

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset if self.evaluate else None,
                compute_metrics=self.compute_metrics,
                data_collator=self.custom_vision_collator,
                **trainer_kwargs,
            )

            # Ensure all model tensors are contiguous
            for param in self.model.parameters():
                param.data = param.data.contiguous()

            # Train the model
            trainer.train()
            trainer.save_model()

            if self.evaluate:
                eval_result = trainer.evaluate()
                self.log.info(f"Evaluation results: {eval_result}")

            # Save the model configuration to Hugging Face Hub if hf_repo_id is not None
            if self.hf_repo_id:
                self.config.save_pretrained(os.path.join(self.output.output_folder, "model"))
                self.upload_to_hf_hub()

        except Exception as e:
            self.log.exception(f"Failed to fine tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise
        self.state.set_state(self.id, {"success": True})

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        metric = evaluate.load("mean_iou")

        with torch.no_grad():
            logits, labels = eval_pred
            logits_tensor = torch.from_numpy(logits)
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_labels = logits_tensor.detach().cpu().numpy()
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(self.model.config.id2label),
                ignore_index=255,
                reduce_labels=False,
            )
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    metrics[key] = value.tolist()
            return metrics
