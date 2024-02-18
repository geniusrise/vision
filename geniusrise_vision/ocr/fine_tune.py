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
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_vision.base import VisionFineTuner
from typing import Union, Dict, Optional, Callable
from PIL import Image
import numpy as np
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import pil_to_tensor
from datasets import load_dataset, load_metric, Dataset, DatasetDict, load_from_disk
from transformers import default_data_collator
from transformers import (
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EvalPrediction,
)


class OCRFineTuner(VisionFineTuner):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs) -> None:
        super().__init__(input=input, output=output, state=state)
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.cer_metric = load_metric("cer")

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
            dataset = self._load_local_dataset(dataset_path, **kwargs)

        if hasattr(self, "map_data") and self.map_data:
            fn = eval(self.map_data)  # type: ignore
            dataset = dataset.map(fn)
        else:
            dataset = dataset

        return dataset

    def _load_local_dataset(self, dataset_path: str, **kwargs) -> TensorDataset:
        images = []
        labels = []  # type: ignore
        dataset = []

        has_train_test_split = all(os.path.isdir(os.path.join(dataset_path, split)) for split in ["train", "test"])
        splits = ["train", "test"] if has_train_test_split else [""]

        for split in splits:
            split_dir = os.path.join(dataset_path, split) if split else dataset_path
            images_dir = os.path.join(split_dir, "images")
            annotations_dir = os.path.join(split_dir, "annotations")

            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                annotation_path = os.path.join(annotations_dir, img_file.replace(".jpg", ".txt"))

                if os.path.isfile(img_path) and os.path.isfile(annotation_path):
                    with open(annotation_path, "r") as file:
                        text = file.read().strip()

                    # Process image
                    image = Image.open(img_path).convert("RGB")
                    image_tensor = pil_to_tensor(image).float()
                    images.append(image_tensor)

                    # Process text using tokenizer
                    label_tensor = self.processor.tokenizer(
                        text, padding="max_length", max_length=128, return_tensors="pt"
                    ).input_ids.squeeze()
                    label_tensor[label_tensor == -100] = self.processor.tokenizer.pad_token_id

                    dataset.append({"pixel_values": image_tensor.squeeze(), "labels": label_tensor})

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

    def fine_tune(
        self,
        model_name: str,
        processor_name: str,
        num_train_epochs: int,
        per_device_batch_size: int,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        device_map: str | dict = "auto",
        compile: bool = False,
        evaluate: bool = False,
        save_steps: int = 500,
        save_total_limit: Optional[int] = None,
        load_best_model_at_end: bool = False,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
        map_data: Optional[Callable] = None,
        use_huggingface_dataset: bool = False,
        huggingface_dataset: str = "",
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: bool = True,
        hf_create_pr: bool = False,
        notification_email: str = "",
        learning_rate: float = 1e-5,
        # extra params
        early_stopping: bool = True,
        max_length: int = 64,
        no_repeat_ngram_size: int = 3,
        length_penalty: float = 2.0,
        num_beams: int = 4,
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
            use_cuda (bool): Flag to utilize CUDA for GPU acceleration.
            precision (str): The floating-point precision to be used by the model. Options are 'float32', 'float16', 'bfloat16'.
            device_map (str | dict, optional): The device map for distributed training. Defaults to "auto".
            evaluate (bool, optional): Whether to evaluate the model after training. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            save_steps (int, optional): Number of steps between checkpoints. Defaults to 500.
            save_total_limit (Optional[int], optional): Maximum number of checkpoints to keep. Older checkpoints are deleted. Defaults to None.
            load_best_model_at_end (bool, optional): Whether to load the best model (according to evaluation) at the end of training. Defaults to False.
            metric_for_best_model (Optional[str], optional): The metric to use to compare models. Defaults to None.
            greater_is_better (Optional[bool], optional): Whether a larger value of the metric indicates a better model. Defaults to None.
            map_data (Callable, optional): A function to map data before training. Defaults to None.
            use_huggingface_dataset (bool, optional): Whether to load a dataset from huggingface hub.
            huggingface_dataset (str, optional): The huggingface dataset to use.
            hf_repo_id (str, optional): The Hugging Face repo ID. Defaults to None.
            hf_commit_message (str, optional): The Hugging Face commit message. Defaults to None.
            hf_token (str, optional): The Hugging Face token. Defaults to None.
            hf_private (bool, optional): Whether to make the repo private. Defaults to True.
            hf_create_pr (bool, optional): Whether to create a pull request. Defaults to False.
            notification_email (str, optional): Whether to notify after job is complete. Defaults to None.
            learning_rate (float, optional): Learning rate for backpropagation.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            None
        """
        try:
            self.model_name = model_name
            self.processor_name = processor_name
            self.num_train_epochs = num_train_epochs
            self.per_device_batch_size = per_device_batch_size
            self.model_class = model_class
            self.processor_class = processor_class
            self.use_cuda = use_cuda
            self.precision = precision
            self.device_map = device_map
            self.compile = compile
            self.evaluate = evaluate
            self.save_steps = save_steps
            self.save_total_limit = save_total_limit
            self.load_best_model_at_end = load_best_model_at_end
            self.metric_for_best_model = metric_for_best_model
            self.greater_is_better = greater_is_better
            self.map_data = map_data
            self.use_huggingface_dataset = use_huggingface_dataset
            self.huggingface_dataset = huggingface_dataset
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr
            self.notification_email = notification_email
            self.learning_rate = learning_rate
            self.early_stopping = early_stopping
            self.max_length = max_length
            self.no_repeat_ngram_size = no_repeat_ngram_size
            self.length_penalty = length_penalty
            self.num_beams = num_beams

            model_kwargs = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}

            self.load_models(
                model_name=self.model_name,
                processor_name=self.processor_name,
                model_class=self.model_class,
                processor_class=self.processor_class,
                device_map=self.device_map,
                precision=self.precision,
                **model_kwargs,
            )

            # Load dataset
            dataset_kwargs = {k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k}
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {k.replace("trainer_", ""): v for k, v in kwargs.items() if "trainer_" in k}
            training_kwargs = {k.replace("training_", ""): v for k, v in kwargs.items() if "training_" in k}

            # set special tokens used for creating the decoder_input_ids from the labels
            self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id  # type: ignore
            self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id  # type: ignore
            self.model.config.vocab_size = self.model.config.decoder.vocab_size  # type: ignore
            self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id  # type: ignore

            # Create training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=os.path.join(self.output.output_folder, "model"),
                predict_with_generate=True,
                evaluation_strategy="steps",
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_batch_size,
                per_device_eval_batch_size=per_device_batch_size,
                save_steps=self.save_steps,
                save_total_limit=self.save_total_limit,
                load_best_model_at_end=self.load_best_model_at_end,
                metric_for_best_model=self.metric_for_best_model,
                greater_is_better=self.greater_is_better,
                dataloader_num_workers=4,
                learning_rate=self.learning_rate,
                **training_kwargs,
            )

            if compile:
                self.model = torch.compile(self.model)  # type: ignore

            # Create trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                tokenizer=self.processor.feature_extractor,
                args=training_args,
                compute_metrics=self.compute_metrics,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset if self.evaluate else None,
                data_collator=default_data_collator,
                **trainer_kwargs,
            )

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

        self.done()

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        labels_ids = eval_pred.label_ids
        pred_ids = eval_pred.predictions
        # Decode predictions
        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        # Replace -100 in labels with pad_token_id and then decode
        labels_ids = np.where(labels_ids == -100, self.processor.tokenizer.pad_token_id, labels_ids)
        label_str = [self.processor.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels_ids]

        cer = self.cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}
