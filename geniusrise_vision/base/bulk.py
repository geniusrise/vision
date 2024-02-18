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

from typing import Any, Dict, Optional, Tuple

import torch
import transformers
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import AutoModel, AutoProcessor
from geniusrise_vision.base.communication import send_email


class VisionBulk(Bolt):
    """
    A class representing the VisionBulk operations that inherits from Bolt.

    This class encapsulates methods for loading and utilizing models for image-related bulk operations.
    It must be subclassed with an implementation for the 'generate' method which is specific to the inheriting class.

    Attributes:
        model (Any): The machine learning model used for processing.
        processor (Any): The processor used for preparing data for the model.

    Args:
        input (BatchInput): A data structure containing input data batches.
        output (BatchOutput): A data structure to store the output data from the model.
        state (State): An object representing the state of the processing operation.
    """

    model: Any
    processor: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
    ):
        """
        Initializes the VisionBulk instance with input data, output data holder, and state.

        Args:
            input (BatchInput): The input data for the bulk operation.
            output (BatchOutput): The holder for the output results of the bulk operation.
            state (State): The state of the bulk operation, containing any required metadata or context.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def generate(*args, **kwargs):
        """
        Generates results based on the loaded models and input data.

        This method is a placeholder and must be implemented by subclasses.
        """
        raise NotImplementedError("Has to be implemented by every inheriting class")

    def _get_torch_dtype(self, precision: str) -> torch.dtype:
        """
        Determines the torch dtype based on the specified precision.

        Args:
            precision (str): The desired precision for computations.

        Returns:
            torch.dtype: The corresponding torch dtype.

        Raises:
            ValueError: If an unsupported precision is specified.
        """
        dtype_map = {
            "float32": torch.float32,
            "float": torch.float,
            "float64": torch.float64,
            "double": torch.double,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "half": torch.half,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "int16": torch.int16,
            "short": torch.short,
            "int32": torch.int32,
            "int": torch.int,
            "int64": torch.int64,
            "quint8": torch.quint8,
            "qint8": torch.qint8,
            "qint32": torch.qint32,
        }
        return dtype_map.get(precision, torch.float)

    def load_models(
        self,
        model_name: str,
        processor_name: str,
        model_revision: Optional[str] = None,
        processor_revision: Optional[str] = None,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = False,
        compile: bool = False,
        flash_attention: bool = False,
        better_transformers: bool = False,
        **model_args: Any,
    ) -> Tuple[AutoModel, AutoProcessor]:
        """
        Loads the model and processor necessary for image processing tasks.

        Args:
            model_name (str): The name of the model to be loaded from Hugging Face's model repository.
            processor_name (str): The name of the processor to be used for preprocessing the data.
            model_revision (Optional[str]): The specific revision of the model to be loaded.
            processor_revision (Optional[str]): The specific revision of the processor to be loaded.
            model_class (str): The class name of the model to be loaded from the transformers package.
            processor_class (str): The class name of the processor to be loaded.
            use_cuda (bool): Flag to utilize CUDA for GPU acceleration.
            precision (str): The floating-point precision to be used by the model. Options are 'float32', 'float16', 'bfloat16'.
            quantization (int): The bit level for model quantization (0 for none, 8 for 8-bit quantization).
            device_map (Union[str, Dict, None]): The device mapping for model parallelism. 'auto' or specific mapping dict.
            max_memory (Dict[int, str]): The maximum memory configuration for the model per device.
            torchscript (bool): Flag to enable TorchScript for model optimization.
            compile (bool): Flag to enable JIT compilation of the model.
            flash_attention (bool): Flag to enable Flash Attention optimization for faster processing.
            better_transformers (bool): Flag to enable Better Transformers optimization for faster processing.
            **model_args (Any): Additional arguments to be passed to the model's 'from_pretrained' method.

        Returns:
            Tuple[AutoModelForCausalLM, AutoProcessor]: A tuple containing the loaded model and processor.
        """
        self.log.info(f"Loading Hugging Face model: {model_name}")

        torch_dtype = self._get_torch_dtype(precision)

        if use_cuda and not device_map:
            device_map = "auto"

        ModelClass = getattr(transformers, model_class)
        processorClass = getattr(transformers, processor_class)

        # Load the model and processor
        processor = processorClass.from_pretrained(processor_name, revision=processor_revision, torch_dtype=torch_dtype)

        self.log.info(f"Loading model from {model_name} {model_revision} with {model_args}")

        if flash_attention:
            model_args = {**model_args, **{"attn_implementation": "flash_attention_2"}}

        if quantization == 8:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                load_in_8bit=True,
                **model_args,
            )
        elif quantization == 4:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                load_in_4bit=True,
                **model_args,
            )
        else:
            model = ModelClass.from_pretrained(
                model_name,
                revision=model_revision,
                torch_dtype=torch_dtype,
                torchscript=torchscript,
                max_memory=max_memory,
                device_map=device_map,
                **model_args,
            )

        if compile and not torchscript:
            model = torch.compile(model)

        # Set to evaluation mode for inference
        model.eval()

        self.log.debug("Hugging Face model and processor loaded successfully.")
        return model, processor

    def done(self):
        if self.notification_email:
            self.output.flush()
            send_email(recipient=self.notification_email, bucket_name=self.output.bucket, prefix=self.output.s3_folder)
