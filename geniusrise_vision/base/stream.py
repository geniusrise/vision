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

from typing import Any, Dict, Optional, Tuple, List, Union
import os
import torch
import transformers
from geniusrise import StreamingInput, StreamingOutput, Bolt, State
from geniusrise.logging import setup_logger
from transformers import AutoModel, AutoProcessor
from uform.gen_model import VLMForCausalLM, VLMProcessor
from optimum.bettertransformer import BetterTransformer
import llama_cpp
from llama_cpp import Llama as LlamaCPP
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class VisionStream(Bolt):
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
        input: StreamingInput,
        output: StreamingOutput,
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
        model_location: Optional[str] = None,
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

        # Note: exception for Uform models
        if "uform" in model_name.lower():
            model = VLMForCausalLM.from_pretrained(model_name)
            processor = VLMProcessor.from_pretrained(processor_name)
            if use_cuda:
                model = model.to(device_map)

            return model, processor

        ModelClass = getattr(transformers, model_class)
        processorClass = getattr(transformers, processor_class)

        # Load the model and processor
        if model_name == "local":
            processor = processorClass.from_pretrained(model_location, torch_dtype=torch_dtype)
        else:
            processor = processorClass.from_pretrained(
                processor_name, revision=processor_revision, torch_dtype=torch_dtype
            )

        self.log.info(f"Loading model from {model_name} {model_revision} with {model_args}")

        if flash_attention:
            model_args = {**model_args, **{"attn_implementation": "flash_attention_2"}}

        if quantization == 8:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
                    load_in_8bit=True,
                    **model_args,
                )
            else:
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
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
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
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
                    load_in_4bit=True,
                    **model_args,
                )
        else:
            if model_name == "local":
                model = ModelClass.from_pretrained(
                    model_location,
                    torch_dtype=torch_dtype,
                    torchscript=torchscript,
                    max_memory=max_memory,
                    device_map=device_map,
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

        if better_transformers:
            model = BetterTransformer.transform(model, keep_original_model=True)

        # Set to evaluation mode for inference
        model.eval()

        self.log.debug("Hugging Face model and processor loaded successfully.")
        return model, processor

    def load_models_llama_cpp(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None,
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        yarn_ext_factor: float = -1.0,
        yarn_attn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_orig_ctx: int = 0,
        mul_mat_q: bool = True,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = True,
        last_n_tokens_size: int = 64,
        lora_base: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_path: Optional[str] = None,
        numa: Union[bool, int] = False,
        chat_format: Optional[str] = None,
        chat_handler: Optional[llama_cpp.llama_chat_format.LlamaChatCompletionHandler] = None,
        draft_model: Optional[llama_cpp.LlamaDraftModel] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[LlamaCPP, Optional[PreTrainedTokenizerBase]]:
        """
        Initializes and loads LLaMA model with llama.cpp backend, along with an optional tokenizer.

        Args:
            model_path (str): Path to the LLaMA model.
            n_gpu_layers (int): Number of layers to offload to GPU. Default is 0.
            split_mode (int): Split mode for distributing model across GPUs.
            main_gpu (int): Main GPU index.
            tensor_split (Optional[List[float]]): Tensor split configuration.
            vocab_only (bool): Whether to load vocabulary only.
            use_mmap (bool): Use memory-mapped files for model loading.
            use_mlock (bool): Lock model data in RAM.
            kv_overrides (Optional[Dict[str, Union[bool, int, float]]]): Key-value pairs for model overrides.
            seed (int): Random seed for initialization.
            n_ctx (int): Number of context tokens.
            n_batch (int): Batch size for processing prompts.
            n_threads (Optional[int]): Number of threads for generation.
            n_threads_batch (Optional[int]): Number of threads for batch processing.
            rope_scaling_type (Optional[int]): RoPE scaling type.
            rope_freq_base (float): Base frequency for RoPE.
            rope_freq_scale (float): Frequency scaling for RoPE.
            yarn_ext_factor (float): YaRN extrapolation mix factor.
            yarn_attn_factor (float): YaRN attention factor.
            yarn_beta_fast (float): YaRN beta fast parameter.
            yarn_beta_slow (float): YaRN beta slow parameter.
            yarn_orig_ctx (int): Original context size for YaRN.
            mul_mat_q (bool): Whether to multiply matrices for queries.
            logits_all (bool): Return logits for all tokens.
            embedding (bool): Enable embedding mode only.
            offload_kqv (bool): Offload K, Q, V matrices to GPU.
            last_n_tokens_size (int): Size for the last_n_tokens buffer.
            lora_base (Optional[str]): Base model path for LoRA.
            lora_scale (float): Scale factor for LoRA adjustments.
            lora_path (Optional[str]): Path to LoRA adjustments.
            numa (Union[bool, int]): NUMA configuration.
            chat_format (Optional[str]): Chat format configuration.
            chat_handler (Optional[llama_cpp.LlamaChatCompletionHandler]): Handler for chat completions.
            draft_model (Optional[llama_cpp.LlamaDraftModel]): Draft model for speculative decoding.
            tokenizer (Optional[PreTrainedTokenizerBase]): Custom tokenizer instance.
            verbose (bool): Enable verbose logging.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[LlamaCPP, Optional[PreTrainedTokenizerBase]]: The loaded LLaMA model and tokenizer.
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")

        self.log.info(f"Loading LLaMA model from {model_path} with llama.cpp backend.")

        llama_model = LlamaCPP(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            split_mode=split_mode,
            main_gpu=main_gpu,
            tensor_split=tensor_split,
            vocab_only=vocab_only,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            kv_overrides=kv_overrides,
            seed=seed,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            rope_scaling_type=rope_scaling_type,
            rope_freq_base=rope_freq_base,
            rope_freq_scale=rope_freq_scale,
            yarn_ext_factor=yarn_ext_factor,
            yarn_attn_factor=yarn_attn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            yarn_orig_ctx=yarn_orig_ctx,
            mul_mat_q=mul_mat_q,
            logits_all=logits_all,
            embedding=embedding,
            offload_kqv=offload_kqv,
            last_n_tokens_size=last_n_tokens_size,
            lora_base=lora_base,
            lora_scale=lora_scale,
            lora_path=lora_path,
            numa=numa,
            chat_format=chat_format,
            chat_handler=chat_handler,
            draft_model=draft_model,
            tokenizer=tokenizer,
            verbose=verbose,
            **kwargs,
        )

        self.log.info("LLaMA model loaded successfully.")

        return llama_model, tokenizer
