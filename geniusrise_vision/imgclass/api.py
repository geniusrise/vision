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
import numpy as np
import json
import torch
import base64
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class VisionClassificationAPI(VisionAPI):
    def __init__(self, model_name: str, input: BatchInput, output: BatchOutput, state: State):
        """
        Initialize the API with a specified model.
        """
        super().__init__(input=input, output=output, state=state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_model(model_name)

    def initialize_model(
        self, 
        model_name: str = None):
        try:
            self.model_name = model_name
            # Initialize model and processor
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            log.info("Model and processor are initialized. API is ready to serve requests.")
        except Exception as e:
            return {"success": False, "error": str(e)}  

    # Copied from transformers.pipelines.text_classification.sigmoid
    def sigmoid(self, _outputs):
        return 1.0 / (1.0 + np.exp(-_outputs))

    # Copied from transformers.pipelines.text_classification.softmax
    def softmax(self, _outputs):
        maxes = np.max(_outputs, axis=-1, keepdims=True)
        shifted_exp = np.exp(_outputs - maxes)
        return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def classify_image(self):
        """
        Endpoint for classifying an image and returning the image with label scores.
        """
        try:
            data = cherrypy.request.json
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs).logits
                outputs = outputs.numpy()

            if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
                scores = self.sigmoid(outputs)
            elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
                scores = self.softmax(outputs)
            else:
                scores = outputs  # No function applied

            # Prepare scores and labels for the response
            labeled_scores = [{"label": self.model.config.id2label[i], "score": float(score)} for i, score in enumerate(scores.flatten())]

            response = {"original_image": image_base64, "predictions": labeled_scores}

            return response

        except Exception as e:
            log.error(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}