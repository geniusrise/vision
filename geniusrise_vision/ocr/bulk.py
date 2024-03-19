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

import json
import os
from PIL import Image
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_vision.base.bulk import VisionBulk
from typing import Optional, Union, List
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from typing import Dict

# from mmocr.apis import MMOCRInferencer
import easyocr
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from .inference import OCRInference


class ImageOCRBulk(VisionBulk, OCRInference):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ) -> None:
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def initialize_model(
        self,
        model_name: str = None,
    ):
        if model_name == "easyocr":
            lang = self.model_args.get("lang", "en")
            self.reader = easyocr.Reader(["ch_sim", lang], quantize=self.quantization)
        # elif model_name == "mmocr":
        #     self.mmocr_infer = MMOCRInferencer(det="dbnet", rec="svtr-small", kie="SDMGR", device=self.device)
        elif model_name == "paddleocr":
            lang = self.model_args.get("lang", "en")
            self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.use_cuda)

    def load_dataset(
        self,
        dataset_path: str,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:
        """
        Load a dataset for image OCR from a local path or Hugging Face Datasets.

        Args:
            dataset_path (Union[str, None], optional): The local path to the dataset directory. Defaults to None.
            **kwargs: Additional arguments.

        Returns:
            Union[Dataset, DatasetDict, Optional[Dataset]]: The loaded dataset.
        """

        if self.use_huggingface_dataset:
            dataset = load_dataset(self.huggingface_dataset)
        elif os.path.isfile(os.path.join(dataset_path, "dataset_info.json")):
            dataset = load_from_disk(dataset_path)
        else:
            dataset = self._load_local_dataset(dataset_path, **kwargs)

        if hasattr(self, "map_data") and self.map_data:
            fn = eval(self.map_data)
            dataset = dataset.map(fn)
        else:
            dataset = dataset

        return dataset

    def _load_local_dataset(self, dataset_path: str) -> Dataset:
        """
        Load a dataset for image classification from a folder containing images and PDFs.

        Args:
            dataset_path (str): The path to the dataset directory.

        Returns:
            Tuple[List[torch.Tensor], List[str]]: A tuple containing a list of image tensors and corresponding file paths.
        """
        # Supported image formats and PDF
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".pdf"]

        # List of all image and PDF file paths in the dataset directory
        file_paths = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in supported_formats]

        # Create lists for images and paths
        images = []
        paths = []

        for file_path in file_paths:
            full_path = os.path.join(dataset_path, file_path)
            try:
                if full_path.lower().endswith(".pdf"):
                    # Convert each page of the PDF to an image
                    pages = convert_from_path(full_path)
                    for page in pages:
                        images.append(page.convert("RGB"))
                        paths.append(full_path)
                else:
                    with Image.open(full_path).convert("RGB") as img:
                        images.append(img)
                    paths.append(full_path)

            except Exception as e:
                self.log.exception(f"Error processing {file_path}: {e}")

        # Convert lists to PyTorch Dataset
        return Dataset.from_dict({"image": images, "path": paths})

    def ocr(
        self,
        model_name,
        model_class: str = "AutoModelForImageClassification",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        device_map: str | Dict | None = "auto",
        max_memory=None,
        torchscript=False,
        compile: bool = False,
        flash_attention: bool = False,
        better_transformers: bool = False,
        batch_size=32,
        use_huggingface_dataset: bool = False,
        huggingface_dataset: str = "",
        use_easyocr_bbox: bool = False,
        **kwargs,
    ) -> None:
        """
        Perform OCR on images using a specified OCR engine.

        Args:
            model_name (str): Name or path of the model.
            model_class (str): Class name of the model (default "AutoModelForImageClassification").
            processor_class (str): Class name of the processor (default "AutoProcessor").
            use_cuda (bool): Whether to use CUDA for model inference (default False).
            precision (str): Precision for model computation (default "float").
            quantization (int): Level of quantization for optimizing model size and speed (default 0).
            device_map (str | Dict | None): Specific device to use for computation (default "auto").
            max_memory (Dict): Maximum memory configuration for devices.
            torchscript (bool, optional): Whether to use a TorchScript-optimized version of the pre-trained language model. Defaults to False.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to True.
            flash_attention (bool): Whether to use flash attention optimization (default False).
            batch_size (int): Number of classifications to process simultaneously (default 32).
            use_huggingface_dataset (bool, optional): Whether to load a dataset from huggingface hub.
            huggingface_dataset (str, optional): The huggingface dataset to use.
            **kwargs: Arbitrary keyword arguments for model and generation configurations.
        """
        self.model_class = model_class
        self.processor_class = processor_class
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.compile = compile
        self.flash_attention = flash_attention
        self.better_transformers = better_transformers
        self.batch_size = batch_size
        self.use_huggingface_dataset = use_huggingface_dataset
        self.huggingface_dataset = huggingface_dataset

        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None
            processor_name = model_name

        # Store model and processor details
        self.model_name = model_name
        self.model_revision = model_revision
        self.processor_name = model_name
        self.processor_revision = processor_revision

        model_args = {k.replace("model_", ""): v for k, v in kwargs.items() if "model_" in k}
        self.model_args = model_args

        if model_name not in ["easyocr", "mmocr", "paddleocr"]:
            self.model, self.processor = self.load_models(
                model_name=self.model_name,
                processor_name=self.processor_name,
                model_revision=self.model_revision,
                processor_revision=self.processor_revision,
                model_class=self.model_class,
                processor_class=self.processor_class,
                use_cuda=self.use_cuda,
                precision=self.precision,
                quantization=self.quantization,
                device_map=self.device_map,
                max_memory=self.max_memory,
                torchscript=self.torchscript,
                compile=self.compile,
                flash_attention=self.flash_attention,
                better_transformers=self.better_transformers,
                **self.model_args,
            )
        else:
            self.initialize_model(self.model_name)

        dataset_path = self.input.input_folder
        self.output_path = self.output.output_folder

        # Load dataset
        self.dataset = self.load_dataset(dataset_path)
        if self.dataset is None:
            self.log.error("Failed to load dataset.")
            return

        for batch_idx in range(0, len(self.dataset["image"]), self.batch_size):
            batch_images = self.dataset["image"][batch_idx : batch_idx + self.batch_size]
            batch_paths = self.dataset["path"][batch_idx : batch_idx + self.batch_size]

            ocr_results = []
            for img_path in batch_paths:
                image = Image.open(img_path)

                if model_name not in ["easyocr", "mmocr", "paddleocr"]:
                    result = self.process_huggingface_models(image, use_easyocr_bbox)
                else:
                    result = self.process_other_models(image)
                ocr_results.append(result)

            self._save_ocr_results(ocr_results, batch_paths, self.output_path, batch_idx)

    def _save_ocr_results(self, ocr_texts: List[str], batch_paths: List[str], output_path: str, batch_index) -> None:
        """
        Save OCR results to a JSON file.

        Args:
            ocr_texts (List[str]): OCR text for each image in the batch.
            batch_paths (List[str]): Paths of the images in the batch.
            output_path (str): Directory to save the OCR results.
            batch_index (int): Index of the current batch.
        """
        # Create a list of dictionaries with image paths and their corresponding OCR texts
        results = [
            {"image_path": image_path, "ocr_text": ocr_text} for image_path, ocr_text in zip(batch_paths, ocr_texts)
        ]
        # Save to JSON
        file_name = os.path.join(self.output_path, f"ocr_results_batch_{batch_index}.json")
        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)

        self.log.info(f"OCR results for batch {batch_index} saved to {file_name}")
