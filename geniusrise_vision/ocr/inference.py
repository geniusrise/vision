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

import numpy as np
import cv2
import re
import easyocr
from PIL import Image
from typing import Any

# from mmocr.apis import MMOCRInferencer
from geniusrise import BatchInput, BatchOutput, State


class _OCRInference:
    def process_huggingface_models(self, image: Image.Image, use_easyocr_bbox: bool):
        """
        Processes the image using a Hugging Face model specified for OCR tasks. Supports advanced configurations
        and post-processing to handle various OCR-related challenges.

        Args:
            image (Image.Image): The image to process.
            use_easyocr_bbox (bool): Whether to use EasyOCR to detect text bounding boxes before processing with
                                     Hugging Face models.

        Returns:
            str: The recognized text from the image.
        """
        # Convert PIL Image to Tensor
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        if self.use_cuda:
            pixel_values = pixel_values.to(self.device_map)

        if "donut" in self.model_name.lower():
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            if self.use_cuda:
                decoder_input_ids = decoder_input_ids.to(self.device_map)

            # Generate transcription using Nougat
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, ""
            )
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            sequence = self.processor.token2json(sequence)

        elif "nougat" in self.model_name.lower():
            # Generate transcription using Nougat
            outputs = self.model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=1024,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            )
            sequence = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            sequence = self.processor.post_process_generation(sequence, fix_markdown=True)
        else:
            if use_easyocr_bbox:
                self._process_with_easyocr_bbox(image, self.use_cuda)
            else:
                outputs = self.model.generate(pixel_values)
                sequence = self.processor.batch_decode(outputs[0], skip_special_tokens=True)[0]

        return sequence

    def process_other_models(self, image: Image.Image) -> Any:
        """
        Processes the image using non-Hugging Face OCR models like EasyOCR or PaddleOCR based on the initialization.

        Args:
            image (Image.Image): The image to process.

        Returns:
            Any: The OCR results which might include text, bounding boxes, and confidence scores depending on the model.

        Raises:
            ValueError: If an invalid or unsupported OCR model is specified.
        """
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
        """
        A helper method to use EasyOCR for detecting text bounding boxes before processing the image with
        a Hugging Face OCR model.

        Args:
            image (Image.Image): The image to process.
            use_cuda (bool): Whether to use GPU acceleration for EasyOCR.

        Returns:
            str: The recognized text from the image after processing it through the specified OCR model.
        """
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


class OCRInference(VisionBulk, _OCRInference):
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


class OCRInferenceStream(VisionBulk, _OCRInference):
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
