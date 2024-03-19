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
import cherrypy
import easyocr
from PIL import Image

# from mmocr.apis import MMOCRInferencer
from paddleocr import PaddleOCR
from geniusrise import BatchInput, BatchOutput, State
from geniusrise_vision.base import VisionAPI
from .inference import OCRInference


class ImageOCRAPI(VisionAPI, OCRInference):
    r"""
    ImageOCRAPI provides Optical Character Recognition (OCR) capabilities for images, leveraging different OCR engines
    like EasyOCR, PaddleOCR, and Hugging Face models tailored for OCR tasks. This API can decode base64-encoded images,
    process them through the chosen OCR engine, and return the recognized text.

    The API supports dynamic selection of OCR engines and configurations based on the provided model name and arguments,
    offering flexibility in processing various languages and image types.

    Attributes:
        Inherits all attributes from the VisionAPI class.

    Methods:
        ocr(self): Processes an uploaded image for OCR and returns the recognized text.

    Example CLI Usage:

    EasyOCR:

    ```bash
    genius ImageOCRAPI rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        listen \
            --args \
                model_name="easyocr" \
                device_map="cuda:0" \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```

    Paddle OCR:

    ```bash
    genius ImageOCRAPI rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        listen \
            --args \
                model_name="paddleocr" \
                device_map="cuda:0" \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```

    Huggingface models:

    ```bash
    genius ImageOCRAPI rise \
        batch \
            --input_folder ./input \
        batch \
            --output_folder ./output \
        none \
        listen \
            --args \
                model_name="facebook/nougat-base" \
                model_class="VisionEncoderDecoderModel" \
                processor_class="NougatProcessor" \
                device_map="cuda:0" \
                use_cuda=True \
                precision="float" \
                quantization=0 \
                max_memory=None \
                torchscript=False \
                compile=False \
                flash_attention=False \
                better_transformers=False \
                endpoint="*" \
                port=3000 \
                cors_domain="http://localhost:3000" \
                username="user" \
                password="password"
    ```
    """

    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs):
        """
        Initializes the ImageOCRAPI with configurations for input, output, state management, and OCR model specifics.

        Args:
            input (BatchInput): Configuration for the input data.
            output (BatchOutput): Configuration for the output data.
            state (State): State management for the API.
            **kwargs: Additional keyword arguments for extended functionality.
        """
        super().__init__(input=input, output=output, state=state)
        self.hf_model = True

    def initialize_model(self, model_name: str):
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
        """
        Endpoint for performing OCR on an uploaded image. It accepts a base64-encoded image, decodes it, preprocesses
        it through the specified OCR model, and returns the recognized text.

        Args:
            None - Expects input through the POST request's JSON body including 'image_base64', 'model_name',
                   and 'use_easyocr_bbox' (optional).

        Returns:
            Dict[str, Any]: A dictionary containing the success status, recognized text ('result'), and the original
            image name ('image_name') if provided.

        Raises:
            Exception: If an error occurs during image processing or OCR.

        Example CURL Request:
        ```bash
        curl -X POST localhost:3000/api/v1/ocr \
            -H "Content-Type: application/json" \
            -d '{"image_base64": "<base64-encoded-image>", "model_name": "easyocr", "use_easyocr_bbox": true}'
        ```

        or

        ```bash
        (base64 -w 0 test_images_ocr/ReceiptSwiss.jpg | awk '{print "{\"image_base64\": \""$0"\", \"max_length\": 1024}"}' > /tmp/image_payload.json)
        curl -X POST http://localhost:3000/api/v1/ocr \
            -H "Content-Type: application/json" \
            -u user:password \
            -d @/tmp/image_payload.json | jq
        ```
        """
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
