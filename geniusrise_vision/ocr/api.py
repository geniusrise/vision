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

import base64
import io
import torch
import numpy as np
import cherrypy
import json
import os
import subprocess
import cv2
import mmocr
import easyocr
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import TrOCRProcessor, NougatProcessor, VisionEncoderDecoderModel
from mmocr.apis import MMOCRInferencer
from paddleocr import PaddleOCR, draw_ocr
from transformers import StoppingCriteria, StoppingCriteriaList
from collections import defaultdict
from geniusrise_vision.ocr.utils import StoppingCriteriaScores
from transformers import StoppingCriteriaList
from geniusrise import BatchInput, BatchOutput, Bolt, State
from geniusrise.logging import setup_logger
from geniusrise_vision.base import VisionAPI
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps


class ImageOCRAPI(VisionAPI):
    def __init__(self, model_name: str, kind: str, input: BatchInput, output: BatchOutput, state: State, use_cuda: bool = True):
        super().__init__(input=input, output=output, state=state)
         # Initialize model and processor
        self.model_name = model_name
        self.kind = kind
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Initialize the model and processor based on the model name
        if self.model_name == "trocr":
            self.processor = TrOCRProcessor.from_pretrained(f"microsoft/trocr-large-{self.kind}")
            self.model = VisionEncoderDecoderModel.from_pretrained(f"microsoft/trocr-large-{self.kind}")
        elif self.model_name == "easyocr":
            self.reader = easyocr.Reader(['ch_sim','en'], quantize=False)
        elif self.model_name == "mmocr":
            self.mmocr_infer = MMOCRInferencer(det='dbnet', rec='svtr-small', kie='SDMGR', device=self.device)
        elif self.model_name == "paddleocr":
            self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=self.use_cuda)
        elif self.model_name == "nougat":
            self.processor = NougatProcessor.from_pretrained("facebook/nougat-base")
            self.model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        else:
            raise ValueError("Invalid OCR engine. Choose 'trocr', 'easyocr', 'mmocr', 'paddleocr', 'nougat'.")

        if self.model_name in ["trocr", "nougat"]:
            self.model.to(self.device)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def ocr(self):
        """
        Perform OCR on a given image using the specified model.
        """
        try:
            data = cherrypy.request.json
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            if self.model_name == "trocr":
                # OCR using trocr and easyocr
                return self._process_trocr(image, self.kind, self.use_cuda)
            elif self.model_name == "nougat":
                # OCR using nougat
                return self._process_nougat(image, self.use_cuda)
            elif self.model_name == "easyocr":
                # OCR using easyocr
                return self._process_easyocr(image, self.use_cuda)
            elif self.model_name == "mmocr":
                # OCR using mmocr
                return self._process_mmocr(image, self.use_cuda)
            elif self.model_name == "paddleocr":
                # OCR using paddleocr
                return self._process_paddleocr(image, self.use_cuda)

        except Exception as e:
            cherrypy.log.error(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}

    def _process_trocr(
        self,
        image: Image.Image,
        kind: str, 
        use_cuda: bool):
        """
            Perform OCR on a single image by detecting text regions with EasyOCR and 
            recognizing text with TROCR.

            Args:
                image (Image.Image): Image to process.
                kind (str): The kind of TROCR model to use ('printed' or 'handwritten').
                use_cuda (bool): Whether to use CUDA for model inference.

            Returns:
                Dict: A dictionary with combined OCR text and the base64 encoded image.
        """
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['ch_sim','en'], quantize=False)

        # Convert PIL Image to OpenCV format for EasyOCR
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Detect text regions using EasyOCR
        results = reader.readtext(open_cv_image, detail=1)
        image_texts = []

        # OCR using TROCR for each detected text region
        for (bbox, _, _) in results:
            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[2])
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(image.width, x_max), min(image.height, y_max)

            if x_max > x_min and y_max > y_min:
                # Crop the detected region from the PIL Image
                text_region = image.crop((x_min, y_min, x_max, y_max))

                # Convert cropped image to Tensor
                pixel_values = self.processor(images=text_region, return_tensors="pt").pixel_values.to(self.device)

                # Perform OCR using TROCR
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                image_texts.append(generated_text)

        full_text = ' '.join(image_texts)

        # Convert the original PIL image to base64 for returning
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"ocr_text": full_text, "image_base64": image_base64}

    def _process_easyocr(self, image: Image.Image, use_cuda: bool):
        """
        Perform OCR on a single image using EasyOCR and return the OCR text and the image.

        Args:
            image (Image.Image): Image to process.
            use_cuda (bool): Whether to use CUDA for EasyOCR.

        Returns:
            Dict: A dictionary with OCR text and the base64 encoded image.
        """

        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        # Perform OCR using EasyOCR
        ocr_results = self.reader.readtext(open_cv_image, detail=0, paragraph=True)
        concatenated_text = ' '.join(ocr_results)

        # Convert the original PIL image to base64 for returning
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"ocr_text": concatenated_text, "image_base64": image_base64}

    def _process_mmocr(self, image: Image.Image, use_cuda: bool):
        """
        Perform OCR on a single image using MMOCR and return the OCR text and the image.

        Args:
            image (Image.Image): Image to process.
            use_cuda (bool): Whether to use CUDA for MMOCR.

        Returns:
            Dict: A dictionary with OCR text and the base64 encoded image.
        """
        
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image) 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Perform OCR using MMOCR
        result = self.mmocr_infer(open_cv_image, save_vis=False)
        predictions = result['predictions']

        # Extract texts and scores
        texts = [pred['rec_texts'] for pred in predictions]
        concatenated_texts = [' '.join(text) for text in texts]
        full_text = ' '.join(concatenated_texts)

        # Convert the original PIL image to base64 for returning
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"ocr_text": full_text, "image_base64": image_base64}

    
    def _process_paddleocr(self, image: Image.Image, use_cuda: bool):
        """
        Perform OCR on a single image using PaddleOCR and return the OCR text and the image.

        Args:
            image (Image.Image): Image to process.
            use_cuda (bool): Whether to use CUDA for PaddleOCR.

        Returns:
            Dict: A dictionary with OCR text and the base64 encoded image.
        """

        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image) 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Perform OCR using PaddleOCR
        result = self.paddleocr_model.ocr(open_cv_image, cls=False)

        # Extract texts
        texts = [line[1][0] for line in result]
        concatenated_text = ' '.join(texts)

        # Convert the original PIL image to base64 for returning
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"ocr_text": concatenated_text, "image_base64": image_base64}

    def _process_nougat(self, image: Image.Image, use_cuda: bool):
        """
        Perform OCR on a single image using Nougat and return the OCR text and the image.

        Args:
            image (Image.Image): Image to process.
            use_cuda (bool): Whether to use CUDA for Nougat.

        Returns:
            Dict: A dictionary with OCR text and the base64 encoded image.
        """

        # Convert PIL Image to Tensor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        # Generate transcription using Nougat
        outputs = self.model.generate(
            pixel_values.to(self.device),
            min_length=1,
            max_length=3584,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )

        sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        sequence = self.processor.post_process_generation(sequence, fix_markdown=False)

        # Convert the original PIL image to base64 for returning
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"ocr_text": sequence, "image_base64": image_base64}