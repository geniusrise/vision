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
import numpy as np
import cherrypy
import torch
import base64
from PIL import Image


class VisionSegmentationAPI(VisionAPI):
    def __init__(self, input: BatchInput, output: BatchOutput, state: State, **kwargs):
        """ """
        super().__init__(input=input, output=output, state=state)

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
            subtask = data.get("subtask", "")

            kwargs = data
            if "image_base64" in kwargs:
                del kwargs["image_base64"]
            if "subtask" in kwargs:
                del kwargs["subtask"]

            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # in some cases we need to feed the tokenized task as an input
            if self.model.config.__class__.__name__ == "OneFormerConfig":
                if subtask is not None:
                    inputs = self.processor(images=[image], return_tensors="pt", task_inputs=[subtask])
            else:
                inputs = self.processor(images=[image], return_tensors="pt")

            target_size = [(image.height, image.width)]
            # inputs["target_size"] = target_size

            if self.use_cuda:
                inputs = inputs.to(self.device_map)

            with torch.no_grad():
                # target_size = inputs.pop("target_size")
                model_outputs = self.model(**inputs)
                # model_outputs["target_size"] = target_size

                # Post-processing
                # Panoptic segmentation
                if subtask == "panoptic":
                    outputs = self.processor.post_process_panoptic_segmentation(
                        model_outputs,
                        target_sizes=target_size,
                    )[0].cpu()

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

                # Instance segmentation
                elif subtask == "instance":
                    outputs = self.processor.post_process_instance_segmentation(
                        model_outputs,
                        target_sizes=target_size,
                    )[0].cpu()

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

                # Semantic segmentation
                elif subtask == "semantic":
                    outputs = self.processor.post_process_semantic_segmentation(
                        model_outputs, target_sizes=target_size
                    )[0]

                    annotation = []
                    segmentation = outputs.cpu().numpy()
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
            self.log.exception(f"Error processing image: {e}")
            raise e
