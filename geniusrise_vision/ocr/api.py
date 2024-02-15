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
import numpy as np
import cherrypy
import cv2
import easyocr
from PIL import Image
from typing import Any

# from mmocr.apis import MMOCRInferencer
from paddleocr import PaddleOCR
from transformers import StoppingCriteriaList
from geniusrise_vision.ocr.utils import StoppingCriteriaScores
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_vision.base import VisionAPI


class ImageOCRAPI(VisionAPI):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs):
        super().__init__(input=input, output=output, state=state)
        self.hf_model = True

    def initialize_model(
        self,
        model_name: str = None,
    ):
        if model_name == "easyocr":
            lang = self.model_args.get("lang", "en")
            self.reader = easyocr.Reader(["ch_sim", lang], quantize=self.quantization)
        # elif model_name == "mmocr":
        #     self.mmocr_infer = MMOCRInferencer(det="dbnet", rec="svtr-small", kie="SDMGR", device=self.device_map)
        elif model_name == "paddleocr":
            lang = self.model_args.get("lang", "en")
            self.paddleocr_model = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.use_cuda)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def ocr(self):
        if not hasattr(self, "model") or not self.model:
            self.hf_model = False
            self.initialize_model(self.model_name)

        try:
            data = cherrypy.request.json
            use_easyocr_bbox = data.get("use_easyocr_bbox", False)
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            image_name = data.get("image_name", "Unnamed Image")
            if self.hf_model:
                result = self.process_huggingface_models(image, use_easyocr_bbox)
            else:
                result = self.process_other_models(image)

            return {"success": True, "result": result, "image_name": image_name}

        except Exception as e:
            cherrypy.log.error(f"Error processing image: {e}")
            raise e

    def process_huggingface_models(self, image: Image.Image, use_easyocr_bbox: bool):
        # Convert PIL Image to Tensor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device_map)
        if "nougat" in self.model_name.lower():
            # Generate transcription using Nougat
            outputs = self.model.generate(
                pixel_values.to(self.device_map),
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
                self._process_with_easyocr_bbox(image, self.use_cuda)
            else:
                outputs = self.model.generate(pixel_values)
                sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]

        return sequence

    def process_other_models(self, image: Image.Image) -> Any:
        # Convert PIL Image to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        if self.model_name == "easyocr":
            # Perform OCR using EasyOCR
            ocr_results = self.reader.readtext(open_cv_image, detail=0, paragraph=True)
            return ocr_results

        # elif self.model_name == "mmocr":
        #     concatenated_text = ""
        #     # Perform OCR using MMOCR
        #     result = self.mmocr_infer(open_cv_image, save_vis=False)
        #     predictions = result["predictions"]
        #     # Extract texts and scores
        #     texts = [pred["rec_texts"] for pred in predictions]
        #     ocr_texts = [" ".join(text) for text in texts]
        #     concatenated_texts = " ".join(ocr_texts)

        elif self.model_name == "paddleocr":
            # Perform OCR using PaddleOCR
            result = self.paddleocr_model.ocr(open_cv_image, cls=False)
            return result
        else:
            raise ValueError("Invalid OCR engine.")

    def _process_with_easyocr_bbox(
        self,
        image: Image.Image,
        use_cuda: bool,
    ):
        # Initialize EasyOCR reader
        reader = easyocr.Reader(["ch_sim", "en"], quantize=False)

        # Convert PIL Image to OpenCV format for EasyOCR
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Detect text regions using EasyOCR
        results = reader.readtext(open_cv_image, detail=1)
        image_texts = []

        # OCR using TROCR for each detected text region
        for bbox, _, _ in results:
            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[2])
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(image.width, x_max), min(image.height, y_max)
            if x_max > x_min and y_max > y_min:
                # Crop the detected region from the PIL Image
                text_region = image.crop((x_min, y_min, x_max, y_max))
                # Convert cropped image to Tensor
                pixel_values = self.processor(images=text_region, return_tensors="pt").pixel_values.to(self.device_map)
                # Perform OCR using TROCR
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                image_texts.append(generated_text)
        full_text = " ".join(image_texts)
        return full_text
