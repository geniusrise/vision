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

import pytest
import cherrypy
import base64
import io
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
from geniusrise import BatchInput, BatchOutput, InMemoryState, State
from geniusrise_vision.imgclass.api import VisionClassificationAPI 
from transformers import AutoModelForImageClassification, AutoProcessor

# Fixtures
@pytest.fixture
def test_image():
    # Create a simple RGB image for testing
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@pytest.fixture
def api_instance():
    input = MagicMock(spec=BatchInput)
    output = MagicMock(spec=BatchOutput)
    state = MagicMock(spec=InMemoryState)

    api = VisionClassificationAPI("microsoft/resnet-50", input, output, state)

    return api

# Test for correct classification output
def test_classification_output(api_instance, test_image):
    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.body.fp = io.BytesIO(test_image)

    # Call classify_image
    response = api_instance.classify_image()

    # Check if there is a 'label_scores' key in the response
    assert 'label_scores' in response, "No label_scores in the response"

    # Check if the label_scores dictionary is not empty
    assert response['label_scores'], "The label_scores dictionary is empty"

# Test to check for invalid image output 
def test_invalid_image_input(api_instance):
    # Create an invalid image input
    invalid_image = b'notanimage'

    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.body.fp = io.BytesIO(invalid_image)

    # Call classify_image
    response = api_instance.classify_image()

    # Check if the response indicates an error
    assert 'error' in response, "No error indicated for invalid image input"


def test_response_structure(api_instance, test_image):
    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.body.fp = io.BytesIO(test_image)

    # Call classify_image
    response = api_instance.classify_image()

    # Check response structure
    assert isinstance(response, dict), "Response is not a dictionary"
    assert 'original_image' in response, "No original_image in the response"
    assert 'label_scores' in response, "No label_scores in the response"


def test_large_image_input(api_instance):
    # Create a large image for testing
    large_img = Image.new('RGB', (4000, 4000), color='blue')
    img_byte_arr = io.BytesIO()
    large_img.save(img_byte_arr, format='JPEG')
    large_image = img_byte_arr.getvalue()

    # Mock CherryPy request
    cherrypy.request = MagicMock()
    cherrypy.request.body.fp = io.BytesIO(large_image)

    # Call classify_image
    response = api_instance.classify_image()

    # Check if the API can handle large images
    assert 'label_scores' in response, "API failed to handle large image"


# def test_specific_model_output(api_instance, test_image):
#     # Mock CherryPy request
#     cherrypy.request = MagicMock()
#     cherrypy.request.body.fp = io.BytesIO(test_image)

#     # Call classify_image
#     response = api_instance.classify_image()

#     # Check for a specific expected output
#     expected_label = "expected_label"
#     assert expected_label in response['label_scores'], f"{expected_label} not in label scores"

