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
import tempfile

import pytest
from geniusrise import BatchInput, BatchOutput, InMemoryState
from PIL import Image

from geniusrise_vision.imgclass.fine_tune import ImageClassificationFineTuner


@pytest.fixture
def image_classification_fine_tuner():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = ImageClassificationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment(input_dir)

    return klass


def setup_test_environment(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)
    # Create a few sample images in different class directories
    class_names = ["class1", "class2"]
    for class_name in class_names:
        class_dir = os.path.join(test_data_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(5):  # Create 5 images per class
            img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(class_dir, f"img{i}.png")
            img.save(img_path)


MODELS_TO_TEST = {
    # fmt: off
    "small": "microsoft/resnet-50",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


def test_load_local_dataset_single_class(image_classification_fine_tuner, model):
    name, model_name = model
    processor_name = model_name
    model_class = "AutoModelForImageClassification"
    processor_class = "AutoProcessor"

    image_classification_fine_tuner.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuner._load_local_dataset(
        image_classification_fine_tuner.input.get(), is_multiclass=False
    )

    assert len(dataset) == 10  # 5 images per class for 2 classes
    for i in range(len(dataset)):
        assert len(dataset[i]["label"]) == 1  # Single label for each sample


# def test_load_local_dataset_multi_class(setup_test_environment):
#     tuner = ImageClassificationFineTuner(...)
#     dataset_path = setup_test_environment
#     dataset = tuner._load_local_dataset(dataset_path, is_multiclass=True)
#     # Assertions to check if dataset is loaded correctly and labels are as expected for multi-class scenario


# def test_compute_metrics():
#     tuner = ImageClassificationFineTuner(...)
#     # Prepare a mock EvalPrediction object with known predictions and labels
#     eval_pred = ...
#     metrics = tuner.compute_metrics(eval_pred)
#     # Assertions to check if the metrics are computed correctly


# def test_fine_tune():
