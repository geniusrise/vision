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
import torch
import cv2
from PIL import Image
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_vision.base.bulk import VisionBulk
from typing import Optional, Union, List
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForVision2Seq
from mmocr.apis import MMOCRInferencer
import cv2
import easyocr
from paddleocr import PaddleOCR
from transformers import StoppingCriteriaList
from geniusrise_vision.ocr.utils import StoppingCriteriaScores
from transformers import StoppingCriteriaList


class ImageOCRBulk(VisionBulk):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        use_cuda: bool = True,
        use_easyocr_bbox: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> None:

        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

        # Initialize model and processor
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.use_easyocr_bbox = use_easyocr_bbox
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def initialize_model(
        self, model_name: str = None, model_type: str = None, kind: str = None, use_easyocr_bbox: bool = False
    ):

        self.use_easyocr_bbox = use_easyocr_bbox
        self.model_name = model_name
        self.model_type = model_type
        self.kind = kind

        # Initialize the model and processor based on the model name
        if model_type == "hf":
            processor_model_id = f"{model_name}-{kind}" if kind else model_name
            self.processor = AutoProcessor.from_pretrained(processor_model_id)
            self.model = AutoModelForVision2Seq.from_pretrained(processor_model_id).to(self.device)
        elif model_name == "easyocr":
            self.reader = easyocr.Reader(["ch_sim", "en"], quantize=False)
            print("easyocr installed")
        elif model_name == "mmocr":
            self.mmocr_infer = MMOCRInferencer(det="dbnet", rec="svtr-small", kie="SDMGR", device=self.device)
        elif model_name == "paddleocr":
            self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=self.use_cuda)
        else:
            raise ValueError("Invalid OCR engine.")

    def load_dataset(
        self,
        dataset_path: Union[str, None] = None,
        hf_dataset: Union[str, None] = None,
        **kwargs,
    ) -> Union[Dataset, DatasetDict, Optional[Dataset]]:

        if dataset_path:
            return self._load_local_dataset(dataset_path, **kwargs)
        elif hf_dataset:
            return load_dataset(hf_dataset, **kwargs)
        else:
            raise ValueError("Either 'dataset_path' or 'hf_dataset' must be provided")

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
                print(f"Error processing {file_path}: {e}")

        # Convert lists to PyTorch Dataset
        return Dataset.from_dict({"image": images, "path": paths})

    def process(self) -> None:
        """
        Perform OCR on images using a specified OCR engine.

        Args:
                kind (str): The kind of text in the images ('printed' or 'handwritten').
                use_cuda (bool): Whether to use CUDA for model inference.
                ocr_engine (str): The OCR engine to use. Options are 'trocr', 'easyocr', 'mmocr', 'paddleocr', 'nougat'.
        """
        try:
            dataset_path = self.input.input_folder
            self.output_path = self.output.output_folder

            # Load dataset
            self.dataset = self.load_dataset(dataset_path)
            if self.dataset is None:
                self.log.error("Failed to load dataset.")
                return

            if self.model_type == "hf":
                self.process_huggingface_models()
            else:
                print("process has been invoked")
                self.process_other_models()
                print("text generated")

        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_huggingface_models(self):
        for batch_idx in range(0, len(self.dataset["image"]), self.batch_size):
            batch_images = self.dataset["image"][batch_idx : batch_idx + self.batch_size]
            batch_paths = self.dataset["path"][batch_idx : batch_idx + self.batch_size]

            ocr_results = []
            if "nougat" in self.model_name.lower():
                for img_path in batch_paths:
                    image = Image.open(img_path)
                    pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(device)
                    # generate transcription
                    outputs = self.model.generate(
                        pixel_values.to(device),
                        min_length=1,
                        max_length=3584,
                        bad_words_ids=[[processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                        output_scores=True,
                        stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                    )
                    sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
                    sequence = self.processor.post_process_generation(sequence, fix_markdown=False)
                    ocr_results.append(sequence)
            else:
                if self.use_easyocr_bbox:
                    self._process_with_easyocr_bbox(batch_paths)
                else:
                    for img_path in batch_paths:
                        image = Image.open(img_path)
                        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(device)
                        outputs = self.model.generate(pixel_values)
                        sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
                        ocr_results.append(sequence)

        # Save OCR results
        self._save_ocr_results(ocr_results, batch_paths, self.output_path, batch_idx)

    def process_other_models(self):
        print("starting process_other_models")
        for batch_idx in range(0, len(self.dataset["image"]), self.batch_size):
            batch_images = self.dataset["image"][batch_idx : batch_idx + self.batch_size]
            batch_paths = self.dataset["path"][batch_idx : batch_idx + self.batch_size]

            ocr_texts = []
            for image_path in batch_paths:
                print("entering loop")
                if self.model_name == "easyocr":
                    print("easyocr instantiated")
                    # OCR process
                    ocr_results = self.reader.readtext(image_path, detail=0, paragraph=True)
                    print(ocr_results)
                    concatenated_text = " ".join(ocr_results)
                    print(concatenated_text)
                    ocr_texts.append(concatenated_text)
                    print(ocr_texts)
                elif self.model_name == "mmocr":
                    print("mmocr instantiated")
                    image = cv2.imread(image_path)
                    if image is None or image.size == 0:
                        print(f"Failed to load image from {image_path}")
                        continue
                    result = self.mmocr_infer(image_path, save_vis=False)
                    print("result generated")
                    predictions = result["predictions"]
                    print("predictions generated")

                    # Extract texts and scores
                    texts = [pred["rec_texts"] for pred in predictions]
                    scores = [pred["rec_scores"] for pred in predictions]

                    # Concatenate texts for each image
                    concatenated_texts = [" ".join(text) for text in texts]
                    ocr_texts.append(" ".join(concatenated_texts))
                    print(ocr_texts)
                elif self.model_name == "paddleocr":
                    print("paddleocr instantiated")
                    result = self.paddleocr_model.ocr(image_path, cls=False)
                    print("result generated")
                    # Extract texts and scores
                    texts = [line[1][0] for line in result]
                    scores = [line[1][1] for line in result]
                    # Concatenate texts for each image
                    concatenated_text = " ".join(texts)
                    ocr_texts.append(concatenated_text)
                    print(ocr_texts)
                else:
                    raise ValueError("Invalid OCR engine.")
            # Save OCR results
            print("going to start ocr results")
            self._save_ocr_results(ocr_texts, batch_paths, self.output_path, batch_idx)

    def _process_with_easyocr_bbox(self, batch_paths):
        # Initialize EasyOCR reader
        reader = easyocr.Reader(["ch_sim", "en"], quantize=False)
        for image_path in batch_paths:
            results = reader.readtext(image_path)
            image = cv2.imread(image_path)
            image_texts = []
            # Recover text regions detected by EasyOCR
            for bbox, _, _ in results:
                # Extract coordinates from bounding box
                x_min, y_min = map(int, bbox[0])
                x_max, y_max = map(int, bbox[2])
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image.shape[1], x_max)
                y_max = min(image.shape[0], y_max)
                # Extract text region
                if x_max > x_min and y_max > y_min:
                    text_region = image[y_min:y_max, x_min:x_max]
                    pixel_values = self.processor(text_region, return_tensors="pt").pixel_values.to(device)
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    image_texts.append(generated_text)

            full_text = " ".join(image_texts)
            ocr_results.append(full_text)

        return ocr_results

    def _save_ocr_results(self, ocr_texts: List[str], batch_paths: List[str], output_path: str, batch_index) -> None:
        """
        Save OCR results to a JSON file.

        Args:
            ocr_texts (List[str]): OCR text for each image in the batch.
            batch_paths (List[str]): Paths of the images in the batch.
            output_path (str): Directory to save the OCR results.
            batch_index (int): Index of the current batch.
        """
        print("_save_ocr_results started")
        # Create a list of dictionaries with image paths and their corresponding OCR texts
        results = [
            {"image_path": image_path, "ocr_text": ocr_text} for image_path, ocr_text in zip(batch_paths, ocr_texts)
        ]
        print(results)
        # Save to JSON
        file_name = os.path.join(self.output_path, f"ocr_results_batch_{batch_index}.json")
        print(file_name)
        with open(file_name, "w") as file:
            json.dump(results, file, indent=4)

        self.log.info(f"OCR results for batch {batch_index} saved to {file_name}")
