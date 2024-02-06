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

import os
import io
import base64
import tempfile
import pytest
import shutil
import json
import cherrypy
from geniusrise import BatchInput, BatchOutput, InMemoryState
from geniusrise_vision.ocr.api import ImageOCRAPI
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import MagicMock
import random
import string

@pytest.fixture
def base64_test_image():
    # Create an image with random text
    img = Image.new('RGB', (384, 384), color='white')
    draw = ImageDraw.Draw(img)
    text = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    draw.text((10, 10), text, fill='black')

    # Convert the image to a Base64 string
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
    return img_base64

@pytest.fixture
def base64_test_image_text():
    # Create an image with random text
    img = Image.new('RGB', (384, 384), color='white')
    draw = ImageDraw.Draw(img)
    text = "ABCDE12345!@"
    draw.text((10, 10), text, fill='black')

    # Convert the image to a Base64 string
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode()
    return img_base64

OCR_ENGINES = ["easyocr","mmocr","paddleocr","trocr","nougat"]

@pytest.fixture(params=OCR_ENGINES)
def ocr_api_instance(tmpdir, request):
    input_dir = os.path.join(tmpdir, "input")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    state = InMemoryState()

    model_name = request.param
    kind = "printed"

    api_instance = ImageOCRAPI(
        model_name=model_name,
        kind=kind,
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
        use_cuda=False
    )

    return api_instance

def test_ocr_engines(ocr_api_instance, base64_test_image):
    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.json = {"image_base64": base64_test_image}

    # Call the OCR method for the current instance
    response = ocr_api_instance.ocr()

    # Assertions
    assert isinstance(response, dict)
    assert 'ocr_text' in response
    assert 'image_base64' in response 

def test_ocr_specific_text(ocr_api_instance, base64_test_image_text):
    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.json = {"image_base64": base64_test_image_text}

    # Call the OCR method for the current instance
    response = ocr_api_instance.ocr()

    # Assertions
    assert isinstance(response, dict)
    assert 'ocr_text' in response
    assert "ABCDE12345!@" in response['ocr_text'], "Specific text not found in OCR response"