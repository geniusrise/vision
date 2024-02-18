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
from geniusrise_vision.base import VisionAPI
import io
import cherrypy
import torch
import base64
from PIL import Image


class VisualQAAPI(VisionAPI):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initialize the VisualQA API with a specified model and its configuration.
        Inherits from VisionAPI to leverage pre-trained models for Visual Question Answering tasks.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def answer_question(self):
        """
        Endpoint for receiving an image with a question and returning the answer based on visual content.

        Processes the request JSON containing an image in base64 format and a question string.
        Utilizes the loaded model for answering questions related to the image.

        Returns:
            Dict[str, Any]: A dictionary containing the question, and the predicted answer.
        """
        try:
            data = cherrypy.request.json
            image_base64 = data.get("image_base64", "")
            question = data.get("question", "")

            if not image_base64 or not question:
                raise ValueError("Both 'image_base64' and 'question' fields are required.")

            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Prepare inputs for the model
            inputs = self.processor(
                text=question, images=image, return_tensors="pt", padding="max_length", truncation=True, max_length=512
            )

            if self.use_cuda:
                inputs = {k: v.to(self.device_map) for k, v in inputs.items()}

            # Model inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                answer_scores = outputs.logits.softmax(dim=-1)
                predicted_answer_index = answer_scores.argmax(-1).item()
                predicted_answer = self.processor.decode(predicted_answer_index)

            response = {"question": question, "answer": predicted_answer}

            return response

        except Exception as e:
            self.log.error(f"Error processing visual question answering task: {e}")
            raise cherrypy.HTTPError(500, "Internal Server Error")
