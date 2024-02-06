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
from transformers import AutoProcessor, AutoModelForVision2Seq, VisionEncoderDecoderModel
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
    def __init__(self, 
        input: BatchInput, 
        output: BatchOutput, 
        state: State, 
        use_cuda: bool = True):

        super().__init__(input=input, output=output, state=state)
         # Initialize model and processor
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def initialize_model(
        self, 
        model_name: str = None, 
        model_type: str = None, 
        kind: str = None, 
        use_easyocr_bbox: bool = False):

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
            self.reader = easyocr.Reader(['ch_sim','en'], quantize=False)
            print("easyocr installed")
        elif model_name == "mmocr":
            self.mmocr_infer = MMOCRInferencer(det='dbnet', rec='svtr-small', kie='SDMGR', device=self.device)
        elif model_name == "paddleocr":
            self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=self.use_cuda)
        else:
            raise ValueError("Invalid OCR engine.")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def ocr(self):
        try:
            data = cherrypy.request.json
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            print("image initialized")
            image_name = data.get("image_name", "Unnamed Image")
            if self.model_type == "hf":
                ocr_text = self.process_huggingface_models(image)
            else:
                print("entering process_other_models")
                ocr_text = self.process_other_models(image)

            return {"success": True, "ocr_text": ocr_text, "image_name": image_name}

        except Exception as e:
            cherrypy.log.error(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}


    def process_huggingface_models(self, image: Image.Image, kind: str, use_cuda: bool, use_easyocr_bbox: bool):
        # Convert PIL Image to Tensor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        if "nougat" in self.model_name.lower():
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
        else:
            if self.use_easyocr_bbox:
                self._process_with_easyocr_bbox(image,use_cuda)
            else:
                outputs = self.model.generate(pixel_values)
                sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        
        return sequence

    def process_other_models(self, image: Image.Image):
        # Convert PIL Image to OpenCV format
        print("process_other_models entered")
        open_cv_image = np.array(image) 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        if self.model_name == "easyocr":
            # Perform OCR using EasyOCR
            ocr_results = self.reader.readtext(open_cv_image, detail=0, paragraph=True)
            concatenated_text = ' '.join(ocr_results)
        elif self.model_name == "mmocr":
            concatenated_text = ''            
            # Perform OCR using MMOCR
            result = self.mmocr_infer(open_cv_image, save_vis=False)
            predictions = result['predictions']
            # Extract texts and scores
            texts = [pred['rec_texts'] for pred in predictions]
            ocr_texts = [' '.join(text) for text in texts]
            concatenated_texts = ' '.join(ocr_texts)
        elif self.model_name == "paddleocr":
            # Perform OCR using PaddleOCR
            result = self.paddleocr_model.ocr(open_cv_image, cls=False)
            # Extract texts
            texts = [line[1][0] for line in result]
            concatenated_text = ' '.join(texts)
        else:
            raise ValueError("Invalid OCR engine.")
        return concatenated_text
    
    def _process_with_easyocr_bbox(self,
        image: Image.Image,
        use_cuda: bool ):
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
        return full_text
    
