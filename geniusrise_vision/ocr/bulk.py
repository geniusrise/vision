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
import subprocess
import torch
import cv2
from PIL import Image
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_vision.base.bulk import ImageBulk
from typing import Dict, Optional, Union, List, Tuple
from datasets import Dataset, DatasetDict
from transformers import TrOCRProcessor, NougatProcessor, VisionEncoderDecoderModel
import mmocr
from mmocr.apis import MMOCRInferencer
import cv2
import easyocr
from paddleocr import PaddleOCR, draw_ocr
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from geniusrise_vision.ocr.utils import StoppingCriteriaScores
from transformers import StoppingCriteriaList


class ImageOCRBulk(ImageBulk):
    def __init__(
        self, 
        input: BatchInput, 
        output: BatchOutput, 
        state: State, 
        **kwargs) -> None:
        
        super().__init__(input, output, state, **kwargs)
        self.log = setup_logger(self.state)

    def load_dataset(
        self,
        dataset_path: Union[str, None] = None,
        hf_dataset: Union[str, None] = None,
        is_multiclass: bool = False,
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
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.pdf']

        # List of all image and PDF file paths in the dataset directory
        file_paths = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in supported_formats]

        # Create lists for images and paths
        images = []
        paths = []

        for file_path in file_paths:
            full_path = os.path.join(dataset_path, file_path)
            try:
                if full_path.lower().endswith('.pdf'):
                    # Convert each page of the PDF to an image
                    pages = convert_from_path(full_path)
                    for page in pages:
                        images.append(page.convert('RGB'))
                        paths.append(full_path)
                else:
                    with Image.open(full_path).convert('RGB') as img:
                        images.append(img)
                    paths.append(full_path)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Convert lists to PyTorch Dataset
        return Dataset.from_dict({"image": images, "path": paths})

    def process(self, kind: str = "printed", use_cuda: bool = True, ocr_engine: str = "trocr") -> None:
        """
        Perform OCR on images using a specified OCR engine.

        Args:
                kind (str): The kind of text in the images ('printed' or 'handwritten').
                use_cuda (bool): Whether to use CUDA for model inference.
                ocr_engine (str): The OCR engine to use. Options are 'trocr', 'easyocr', 'mmocr', 'paddleocr', 'nougat'.
        """
        if ocr_engine == "trocr":
            self._process_trocr(kind, use_cuda)
        elif ocr_engine == "easyocr":
            self._process_easyocr(use_cuda)
        elif ocr_engine == "mmocr":
            self._process_mmocr(use_cuda)
        elif ocr_engine == "paddleocr":
            self._process_paddleocr(use_cuda)
        elif ocr_engine == "nougat":
            self._process_nougat(use_cuda)
        else:
            raise ValueError("Invalid OCR engine. Choose 'trocr', 'easyocr', 'mmocr', 'paddleocr', 'nougat'.")

    def _process_trocr(
        self, 
        kind: str, 
        use_cuda: bool):
        """
        📖 Perform OCR on images in the input folder and save the OCR results as JSON files in the output folder.

        This method iterates through each image file in `input.input_folder`, performs OCR using the TROCR model,
        and saves the OCR results as JSON files in `output.output_folder`.

        Args:
            kind (str): The kind of TROCR model to use. Default is "printed". Options are "printed" or "handwritten".
            use_cuda (bool): Whether to use CUDA for model inference. Default is True.
        """

        processor = TrOCRProcessor.from_pretrained(f"microsoft/trocr-large-{kind}")
        model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/trocr-large-{kind}")

        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        model.to(device)

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['ch_sim','en'], quantize=False)

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        # Process and OCR in batches
        batch_size = 32
        for batch_idx in range(0, len(dataset['image']), batch_size):
            batch_images = dataset['image'][batch_idx:batch_idx + batch_size]
            batch_paths = dataset['path'][batch_idx:batch_idx + batch_size]

            ocr_results = []
            for image_path in batch_paths:
                results = reader.readtext(image_path)
                image = cv2.imread(image_path)
                image_texts = []

                # Recover text regions detected by EasyOCR
                for (bbox, _, _) in results:
                    # Extract coordinates from bounding box
                    x_min, y_min = map(int, bbox[0])
                    x_max, y_max = map(int, bbox[2])
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(image.shape[1], x_max)
                    y_max = min(image.shape[0], y_max)
                    #Extract text region
                    if x_max>x_min and y_max>y_min :
                        text_region = image[y_min:y_max, x_min:x_max]
                        pixel_values = processor(text_region, return_tensors="pt").pixel_values.to(device)
                        generated_ids = model.generate(pixel_values)
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                        image_texts.append(generated_text)
                
                full_text = ' '.join(image_texts)
                ocr_results.append(full_text)   

            # Save OCR results
            self._save_ocr_results(ocr_results, batch_paths, output_path, batch_idx)

    def _process_easyocr(self, use_cuda: bool):
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['ch_sim','en'], quantize=False)

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        # Process and OCR in batches
        batch_size = 32
        for batch_idx in range(0, len(dataset['image']), batch_size):
            batch_images = dataset['image'][batch_idx:batch_idx + batch_size]
            batch_paths = dataset['path'][batch_idx:batch_idx + batch_size]

            ocr_texts = []
            for image_path in batch_paths:
                # OCR process
                ocr_results = reader.readtext(image_path, detail=0, paragraph=True)
                concatenated_text = ' '.join(ocr_results)
                ocr_texts.append(concatenated_text)

            # Save OCR results
            self._save_ocr_results(ocr_texts, batch_paths, output_path, batch_idx)

    def _process_mmocr(self, use_cuda: bool):
        # Setup MMOCR
        device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
        mmocr_infer = MMOCRInferencer(det='dbnet', rec='svtr-small', kie='SDMGR', device=device)
        dataset_path = self.input.input_folder
        output_path = self.output.output_folder
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        # Process and OCR in batches
        batch_size = 32
        for batch_idx in range(0, len(dataset['image']), batch_size):
            batch_images = dataset['image'][batch_idx:batch_idx + batch_size]
            batch_paths = dataset['path'][batch_idx:batch_idx + batch_size]

            ocr_results = []
            for img_path in batch_paths:
                print(img_path)
                image = cv2.imread(img_path)
                if image is None or image.size == 0:
                    print(f"Failed to load image from {img_path}")
                    continue 
                result = mmocr_infer(img_path, save_vis=False)
                predictions = result['predictions']

                # Extract texts and scores
                texts = [pred['rec_texts'] for pred in predictions]
                scores = [pred['rec_scores'] for pred in predictions]

                # Concatenate texts for each image
                concatenated_texts = [' '.join(text) for text in texts]
                ocr_results.append(' '.join(concatenated_texts))

            # Save OCR results
            self._save_ocr_results(ocr_results, batch_paths, output_path, batch_idx)
    
    def _process_paddleocr(self, use_cuda: bool):
        # Setup PaddleOCR
        device = 'gpu' if use_cuda and torch.cuda.is_available() else 'cpu'
        paddleocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_cuda)

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset= self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        # Process and OCR in batches
        batch_size = 32

        for batch_idx in range(0, len(dataset['image']), batch_size):
            batch_images = dataset['image'][batch_idx:batch_idx + batch_size]
            batch_paths = dataset['path'][batch_idx:batch_idx + batch_size]

            ocr_results = []
            for img_path in batch_paths:
                result = paddleocr_model.ocr(img_path, cls=False)

                # Extract texts and scores
                texts = [line[1][0] for line in result]
                scores = [line[1][1] for line in result]

                # Concatenate texts for each image
                concatenated_text = ' '.join(texts)
                ocr_results.append(concatenated_text)

            # Save OCR results
            self._save_ocr_results(ocr_results, batch_paths, output_path, batch_idx)

    def _process_nougat(self, use_cuda: bool):
         # Initialize Nougat Processor and Model
        processor = NougatProcessor.from_pretrained("facebook/nougat-base")
        model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

        device = 'gpu' if use_cuda and torch.cuda.is_available() else 'cpu'
        model.to(device)

        dataset_path = self.input.input_folder
        output_path = self.output.output_folder

        # Load dataset
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            self.log.error("Failed to load dataset.")
            return

        # Process and OCR in batches
        batch_size = 32

        for batch_idx in range(0, len(dataset['image']), batch_size):
            batch_images = dataset['image'][batch_idx:batch_idx + batch_size]
            batch_paths = dataset['path'][batch_idx:batch_idx + batch_size]

            ocr_results = []
            for img_path in batch_paths:
                image = Image.open(img_path)
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

                # generate transcription
                outputs = model.generate(
                            pixel_values.to(device),
                            min_length=1,
                            max_length=3584,
                            bad_words_ids=[[processor.tokenizer.unk_token_id]],
                            return_dict_in_generate=True,
                            output_scores=True,
                            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
                        )
    
                sequence = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
                sequence = processor.post_process_generation(sequence, fix_markdown=False)

                ocr_results.append(sequence)

        # Save OCR results
        self._save_ocr_results(ocr_results, batch_paths, output_path, batch_idx)
        

    def _save_ocr_results(
        self, 
        ocr_texts: List[str],
        batch_paths: List[str], 
        output_path: str, 
        batch_index) -> None:
        """
        Save OCR results to a JSON file.

        Args:
            ocr_texts (List[str]): OCR text for each image in the batch.
            batch_paths (List[str]): Paths of the images in the batch.
            output_path (str): Directory to save the OCR results.
            batch_index (int): Index of the current batch.
        """
        # Create a list of dictionaries with image paths and their corresponding OCR texts
        results = [{"image_path": image_path, "ocr_text": ocr_text} for image_path, ocr_text in zip(batch_paths, ocr_texts)]

        # Save to JSON
        file_name = os.path.join(output_path, f"ocr_results_batch_{batch_index}.json")
        with open(file_name, 'w') as file:
            json.dump(results, file, indent=4)

        self.log.info(f"OCR results for batch {batch_index} saved to {file_name}")
