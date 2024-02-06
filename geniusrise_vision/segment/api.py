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
import numpy as np
from PIL import Image
import cherrypy
import torch
import base64
from PIL import Image
from transformers import AutoModelForUniversalSegmentation, AutoModelForSemanticSegmentation, AutoImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class VisionSegmentationAPI(VisionAPI):
    def __init__(self, model_name: str, subtask: str, input: BatchInput, output: BatchOutput, state: State):
        """
        Initialize the API with a specified model.
        """
        super().__init__(input=input, output=output, state=state)
         # Initialize model and processor
        self.model_name = model_name
        self.subtask = subtask
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if subtask in {"panoptic", None} and hasattr(self.processor, "post_process_panoptic_segmentation") or model_name in {"facebook/mask2former-swin-small-ade-semantic", "facebook/maskformer-swin-base-ade"}:
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
        elif subtask in {"instance", None} and hasattr(self.processor, "post_process_instance_segmentation"):
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_name)
        elif subtask in {"semantic", None} and hasattr(self.processor, "post_process_semantic_segmentation"):
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        else:
            raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")

        self.model.to(self.device)

        log.info("Model and processor are initialized. API is ready to serve requests.")

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    @cherrypy.tools.allow(methods=["POST"])
    def segment_image(self):
        """
        Endpoint for segmenting an image and returning the imasks and labels.
        """
        try:
            data = cherrypy.request.json
            image_base64 = data.get("image_base64", "")
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            target_size = [(image.height, image.width)]
            if self.model.config.__class__.__name__ == "OneFormerConfig":
                if subtask is None:
                    kwargs = {}
                else:
                    kwargs = {"task_inputs": [subtask]}
                    inputs = self.processor(images=[image], return_tensors="pt", **kwargs)
                    inputs["task_inputs"] = self.tokenizer(
                        inputs["task_inputs"],
                        padding="max_length",
                        max_length=self.model.config.task_seq_len,
                        return_tensors=self.framework,
                    )["input_ids"]
            else:
                inputs = self.processor(images=[image], return_tensors="pt")
            inputs["target_size"] = target_size


            with torch.no_grad():
                target_size = inputs.pop("target_size")
                model_outputs = self.model(**inputs)
                model_outputs["target_size"] = target_size

                fn = None
                if self.subtask in {"panoptic", None} and hasattr(self.processor, "post_process_panoptic_segmentation"):
                    fn = self.processor.post_process_panoptic_segmentation
                elif self.subtask in {"instance", None} and hasattr(self.processor, "post_process_instance_segmentation"):
                    fn = self.processor.post_process_instance_segmentation

                if fn is not None:
                    outputs = fn(
                        model_outputs,
                        target_sizes=model_outputs["target_size"],
                    )[0]

                    annotation = []
                    segmentation = outputs["segmentation"]

                    for segment in outputs["segments_info"]:
                        mask = (segmentation == segment["id"]) * 255
                        mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
                        label = self.model.config.id2label[segment["label_id"]]
                        score = segment["score"]

                        # Convert the original PIL image to base64 for returning
                        buffered = io.BytesIO()
                        mask.save(buffered, format="JPEG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        annotation.append({"score": score, "label": label, "mask": image_base64})

                elif self.subtask in {"semantic", None} and hasattr(self.processor, "post_process_semantic_segmentation"):
                    outputs = self.processor.post_process_semantic_segmentation(
                        model_outputs, target_sizes=model_outputs["target_size"]
                    )[0]

                    annotation = []
                    segmentation = outputs.numpy()
                    labels = np.unique(segmentation)

                    for label in labels:
                        mask = (segmentation == label) * 255
                        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                        label = self.model.config.id2label[label]

                        # Convert the original PIL image to base64 for returning
                        buffered = io.BytesIO()
                        mask.save(buffered, format="JPEG")
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        annotation.append({"score": None, "label": label, "mask": image_base64})
                else:
                    raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")
            return annotation

        except Exception as e:
            log.error(f"Error processing image: {e}")
            return {"success": False, "error": str(e)}