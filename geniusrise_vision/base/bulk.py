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
import transformers
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import (
    AutoProcessor,
    AutoModel,
)


class ImageBulk(Bolt):
    """
    A class representing the ImageBulk operations that inherits from Bolt.

    This class encapsulates methods for loading and utilizing models for image-related bulk operations.
    It must be subclassed with an implementation for the 'generate' method which is specific to the inheriting class.

    Attributes:
        model (Any): The machine learning model used for processing.
        processor (Any): The processor used for preparing data for the model.
        log (Logger): A logger instance for logging information.

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
        Initializes the ImageBulk instance with input data, output data holder, and state.

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

    def load_models(
        self,
        model_name: str,
        processor_name: str,
        model_revision: Optional[str] = None,
        processor_revision: Optional[str] = None,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
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
            use_cuda (bool): Whether to load the model on CUDA-enabled devices.
            precision (str): The floating-point precision to be used by the model. Options are 'float32', 'float16', 'bfloat16'.
            quantization (int): The level of model quantization for memory optimization. Options are 0, 8, 4.
            device_map (Union[str, Dict, None]): The device mapping for model parallelism. 'auto' or specific mapping dict.
            max_memory (Dict[int, str]): The maximum memory configuration for the model per device.
            torchscript (bool): Whether to load the model for TorchScript compatibility.
            **model_args (Any): Additional arguments to be passed to the model's 'from_pretrained' method.

        Returns:
            Tuple[AutoModelForCausalLM, AutoProcessor]: A tuple containing the loaded model and processor.

        Raises:
            ValueError: If an unsupported precision is specified.

        Usage:
            >> image_bulk = ImageBulk(input, output, state)
            >> model, processor = image_bulk.load_models(model_name='model-name', processor_name='processor-name')
        """
        self.log.info(f"Loading Hugging Face model: {model_name}")

        if use_cuda and not device_map:
            device_map = "auto"

        ModelClass = getattr(transformers, model_class)
        processorClass = getattr(transformers, processor_class)

        # Load the model and processor
        processor = processorClass.from_pretrained(
            processor_name, revision=processor_revision
        )

        self.log.info(
            f"Loading model from {model_name} {model_revision} with {model_args}"
        )

        model = ModelClass.from_pretrained(
            model_name,
            revision=model_revision,
            torchscript=torchscript,
            max_memory=max_memory,
            device_map=device_map,
            **model_args,
        )

        # Set to evaluation mode for inference
        model.eval()

        self.log.debug("Hugging Face model and processor loaded successfully.")
        return model, processor
