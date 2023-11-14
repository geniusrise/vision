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
from abc import abstractmethod
from typing import Dict, Optional, Callable

import numpy as np
from datasets import Dataset, DatasetDict
from geniusrise import BatchInput, BatchOutput, Bolt, State
from sklearn.metrics import accuracy_score
from transformers import EvalPrediction, Trainer, TrainingArguments, AutoConfig
from geniusrise.logging import setup_logger


class VisionFineTuner(Bolt):
    """
    A bolt for fine-tuning Hugging Face vision models.

    This bolt uses the Hugging Face Transformers library to fine-tune a pre-trained vision model.
    It uses the `Trainer` class from the Transformers library to handle the training.
    """

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        """
        Initialize the bolt.

        Args:
            input (BatchInput): The batch input data.
            output (BatchOutput): The output data.
            state (State): The state manager.
            evaluate (bool, optional): Whether to evaluate the model. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state)
        self.input = input
        self.output = output
        self.state = state

        self.log = setup_logger(self)

    @abstractmethod
    def load_dataset(
        self, dataset_path: str, **kwargs
    ) -> Dataset | DatasetDict | Optional[Dataset]:
        """
        Load a dataset from a file.

        Args:
            dataset_path (str): The path to the dataset file.
            split (str, optional): The split to load. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the `load_dataset` method.

        Returns:
            Union[Dataset, DatasetDict, None]: The loaded dataset.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def preprocess_data(self, **kwargs):
        """Load and preprocess the dataset"""
        try:
            self.input.copy_from_remote()
            train_dataset_path = os.path.join(self.input.get(), "train")
            eval_dataset_path = os.path.join(self.input.get(), "test")
            self.train_dataset = self.load_dataset(train_dataset_path, **kwargs)
            if self.evaluate:
                self.eval_dataset = self.load_dataset(eval_dataset_path, **kwargs)
        except Exception as e:
            self.log.exception(f"Failed to preprocess data: {e}")
            raise e

    def load_models(
        self,
        model_name: str,
        processor_name: str,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        device_map: str | dict = "auto",
        **kwargs,
    ):
        """
        Load the model and processor.

        Args:
            model_name (str): The name of the model to be loaded.
            processor_name (str, optional): The name of the processor to be loaded. Defaults to None.
            model_class (str, optional): The class of the model. Defaults to "AutoModel".
            processor_class (str, optional): The class of the processor. Defaults to "Autoprocessor".
            device (Union[str, torch.device], optional): The device to be used. Defaults to "cuda".
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If an unsupported precision is chosen.

        Returns:
            None
        """
        try:
            if ":" in model_name:
                model_revision = model_name.split(":")[1]
                model_name = model_name.split(":")[0]
            else:
                model_revision = None
            self.model_name = model_name
            self.log.info(f"Loading model {model_name} and branch {model_revision}")

            # Load model

            # Use AutoConfig to automatically load the configuration
            if self.model_name.lower() == "local":  # type: ignore
                self.log.info(f"Loading local model {model_class} : {self.input.get()}")
                self.config = AutoConfig.from_pretrained(
                    os.path.join(self.input.get(), "/model")
                )
                self.model = getattr(
                    __import__("transformers"), str(model_class)
                ).from_pretrained(
                    os.path.join(self.input.get(), "/model"),
                    device_map=device_map,
                    config=self.config,
                    **kwargs,
                )
            else:
                self.log.info(
                    f"Loading from huggingface hub: {model_class} : {model_name}"
                )
                self.config = AutoConfig.from_pretrained(self.model_name)
                self.model = getattr(
                    __import__("transformers"), str(model_class)
                ).from_pretrained(
                    model_name,
                    revision=model_revision,
                    device_map=device_map,
                    config=self.config,
                    **kwargs,
                )

            # Load processor

            if ":" in processor_name:
                processor_revision = processor_name.split(":")[1]
                processor_name = processor_name.split(":")[0]
            else:
                processor_revision = None
            self.processor_name = processor_name

            if processor_name.lower() == "local":  # type: ignore
                self.log.info(
                    f"Loading local processor : {processor_class} : {self.input.get()}"
                )
                self.processor = getattr(
                    __import__("transformers"), str(processor_class)
                ).from_pretrained(os.path.join(self.input.get(), "/model"))
            else:
                self.log.info(
                    f"Loading processor from huggingface hub: {processor_class} : {processor_name}"
                )
                self.processor = getattr(
                    __import__("transformers"), str(processor_class)
                ).from_pretrained(processor_name, revision=processor_revision)
        except Exception as e:
            self.log.exception(f"Failed to load model: {e}")
            raise

    def upload_to_hf_hub(
        self,
        hf_repo_id: Optional[str] = None,
        hf_commit_message: Optional[str] = None,
        hf_token: Optional[str] = None,
        hf_private: Optional[str] = None,
        hf_create_pr: Optional[str] = None,
    ):
        """Upload the model and processor to Hugging Face Hub."""
        try:
            if self.model:
                self.model.to("cpu").push_to_hub(
                    repo_id=hf_repo_id if hf_repo_id else self.hf_repo_id,
                    commit_message=hf_commit_message
                    if hf_commit_message
                    else self.hf_commit_message,
                    token=hf_token if hf_token else self.hf_token,
                    private=hf_private if hf_private else self.hf_private,
                    create_pr=hf_create_pr if hf_create_pr else self.hf_create_pr,
                )
            if self.processor:
                self.processor.push_to_hub(
                    repo_id=hf_repo_id if hf_repo_id else self.hf_repo_id,
                    commit_message=hf_commit_message
                    if hf_commit_message
                    else self.hf_commit_message,
                    token=hf_token if hf_token else self.hf_token,
                    private=hf_private if hf_private else self.hf_private,
                    create_pr=hf_create_pr if hf_create_pr else self.hf_create_pr,
                )
        except Exception as e:
            self.log.exception(f"Failed to upload model to huggingface hub: {e}")
            raise

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for evaluation. This class implements a simple image classification evaluation.

        Args:
            eval_pred (EvalPrediction): The evaluation predictions.

        Returns:
            dict: The computed metrics.
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
        }

    def fine_tune(
        self,
        model_name: str,
        processor_name: str,
        num_train_epochs: int,
        per_device_batch_size: int,
        model_class: str = "AutoModel",
        processor_class: str = "Autoprocessor",
        device_map: str | dict = "auto",
        evaluate: bool = False,
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
            self.hf_repo_id = hf_repo_id
            self.hf_commit_message = hf_commit_message
            self.hf_token = hf_token
            self.hf_private = hf_private
            self.hf_create_pr = hf_create_pr
            self.map_data = map_data

            model_kwargs = {
                k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k
            }

            self.load_models(
                model_name=self.model_name,
                processor_name=self.processor_name,
                model_class=self.model_class,
                processor_class=self.processor_class,
                device_map=self.device_map,
                **model_kwargs,
            )

            # Load dataset
            dataset_kwargs = {
                k.replace("data_", ""): v for k, v in kwargs.items() if "data_" in k
            }
            self.preprocess_data(**dataset_kwargs)

            # Separate training and evaluation arguments
            trainer_kwargs = {
                k.replace("trainer_", ""): v
                for k, v in kwargs.items()
                if "trainer_" in k
            }
            training_kwargs = {
                k.replace("training_", ""): v
                for k, v in kwargs.items()
                if "training_" in k
            }

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
                self.config.save_pretrained(
                    os.path.join(self.output.output_folder, "model")
                )
                self.upload_to_hf_hub()

        except Exception as e:
            self.log.exception(f"Failed to fine tune model: {e}")
            self.state.set_state(self.id, {"success": False, "exception": str(e)})
            raise
        self.state.set_state(self.id, {"success": True})
