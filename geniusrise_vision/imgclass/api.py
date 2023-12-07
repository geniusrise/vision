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

from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
from geniusrise_vision.base import VisionAPI
import io
import logging
import cherrypy
import torch
import base64
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class VisionClassificationAPI(VisionAPI):
    def __init__(self, model_name, input: BatchInput, output: BatchOutput, state: State):
        """
        Initialize the API with a specified model.
        """
        super().__init__(input=input, output=output, state=state)
         # Initialize model and processor
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        log.info("Model and processor are initialized. API is ready to serve requests.")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify_image(self):
        """
        Endpoint for classifying an image and returning the image with label scores.
        """
        try:
            # Retrieve and process the image from the request
            uploaded_file = cherrypy.request.body.fp
            with Image.open(io.BytesIO(uploaded_file.read())).convert("RGB") as image:
                # Encode original image to base64 for JSON response
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()

                # Preprocess the image
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Perform inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.nn.functional.softmax(logits, dim=-1)

                # Convert results to readable format
                probabilities = probabilities.cpu().numpy().tolist()[0]
                labels = self.model.config.id2label
                label_scores = {labels[i]: prob for i, prob in enumerate(probabilities)}

                # Combine image and label scores in JSON response
                response = {
                    "original_image": img_base64,
                    "label_scores": label_scores
                }

                return response

        except Exception as e:
            log.error(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}