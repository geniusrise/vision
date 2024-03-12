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

import torch
import numpy as np
import BytesIO
from geniusrise import BatchInput, BatchOutput, State, StreamingInput, StreamingOutput
from geniusrise_vision.base.bulk import VisionBulk
from PIL import Image
from typing import Dict, List, Any


class _ImageClassificationInference:
    model: Any
    processor: Any
    use_cuda: bool
    device_map: str | Dict | None

    def sigmoid(self, _outputs):
        """
        Apply the sigmoid function to batched outputs.
        """
        return 1.0 / (1.0 + np.exp(-_outputs))

    def softmax(self, _outputs):
        """
        Apply the softmax function to batched outputs.

        Args:
            _outputs (np.ndarray): Model logits of shape (batch_size, num_classes).

        Returns:
            np.ndarray: Softmax scores of shape (batch_size, num_classes).
        """
        # Ensure that the subtraction for numerical stability and softmax computation are done across the correct axis.
        maxes = np.max(_outputs, axis=1, keepdims=True)
        shifted_exp = np.exp(_outputs - maxes)
        return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

    def classify_one(self, image: BytesIO, **generation_params):
        # Preprocess the image
        inputs = self.processor(images=image, return_tensors="pt")

        if self.use_cuda:
            inputs = {k: v.to(self.device_map) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs, **generation_params).logits
            outputs = outputs.cpu().numpy()

        if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
            scores = self.sigmoid(outputs)
        elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
            scores = self.softmax(outputs)
        else:
            scores = outputs  # No function applied

        # Prepare scores and labels for the response
        labeled_scores: List[Dict[str, Any]] = [
            {"label": self.model.config.id2label[i], "score": float(score)} for i, score in enumerate(scores.flatten())
        ]
        return labeled_scores

    def classify_batch(
        self,
        batch_paths: List[str],
        **generation_params,
    ) -> List[dict]:
        # Process and classify in batches

        batch_inputs = self.processor(
            images=[Image.open(img_path) for img_path in batch_paths], return_tensors="pt", padding=True
        )
        if self.use_cuda:
            batch_inputs = {k: v.to(self.device_map) for k, v in batch_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**batch_inputs, **generation_params).logits

        # Apply appropriate post-processing based on the problem type
        if self.model.config.problem_type == "multi_label_classification" or self.model.config.num_labels == 1:
            scores = self.sigmoid(outputs.cpu().numpy())
        elif self.model.config.problem_type == "single_label_classification" or self.model.config.num_labels > 1:
            scores = self.softmax(outputs.cpu().numpy())
        else:
            scores = outputs.cpu().numpy()  # No function applied

        # Process each item in the batch
        scores = []
        for idx, img_path in enumerate(batch_paths):
            labeled_scores = [
                {"label": self.model.config.id2label[i], "score": float(score)}
                for i, score in enumerate(scores[idx].flatten())
            ]
            scores.append(labeled_scores)
        return scores


class ImageClassificationInference(VisionBulk, _ImageClassificationInference):
    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the SpeechToTextAPI with configurations for speech-to-text processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)


class ImageClassificationInferenceStream(VisionBulk, _ImageClassificationInference):
    def __init__(
        self,
        input: StreamingInput,
        output: StreamingOutput,
        state: State,
        **kwargs,
    ):
        """
        Initializes the SpeechToTextAPI with configurations for speech-to-text processing.

        Args:
            input (BatchInput): The input data configuration.
            output (BatchOutput): The output data configuration.
            state (State): The state configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(input=input, output=output, state=state, **kwargs)
