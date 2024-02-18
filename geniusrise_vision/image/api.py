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
import cherrypy
import torch
import base64
from PIL import Image
import io


class ImageGenerationAPI(VisionAPI):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initialize the ImageGeneration API with a specified model and its configuration.
        Inherits from VisionAPI to leverage pre-trained models for image generation tasks, such as using SDXL.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def generate_image(self):
        """
        Endpoint for receiving a text prompt and returning a generated image based on that prompt.

        Processes the request JSON containing a text prompt.
        Utilizes the loaded model for generating an image based on the text prompt.

        Returns:
            Dict[str, Any]: A dictionary containing the base64-encoded image generated from the text prompt.
        """
        try:
            data = cherrypy.request.json
            prompt = data.get("prompt", "")

            if not prompt:
                raise ValueError("The 'prompt' field is required.")

            # Model inference
            inputs = self.processor(
                text=prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512
            )

            if self.use_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs)
                generated_image_tensor = outputs.images[0]  # Assuming the model returns an object with an images list

            # Convert tensor to PIL Image
            generated_image = Image.fromarray(generated_image_tensor.cpu().numpy().astype("uint8"), "RGB")

            # Convert PIL Image to base64 for transmission
            buffered = io.BytesIO()
            generated_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            response = {"generated_image_base64": img_str}

            return response

        except Exception as e:
            self.log.error(f"Error processing image generation task: {e}")
            raise cherrypy.HTTPError(500, "Internal Server Error")
