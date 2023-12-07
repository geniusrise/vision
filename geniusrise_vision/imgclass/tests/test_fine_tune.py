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
from transformers import EvalPrediction
import numpy as np
import pytest
from geniusrise import BatchInput, BatchOutput, InMemoryState
from PIL import Image

from geniusrise_vision.imgclass.fine_tune import ImageClassificationFineTuner

MODELS_TO_TEST = {
    # fmt: off
    "small": "microsoft/resnet-50",
    # fmt: on
}


# Fixture for models
@pytest.fixture(params=MODELS_TO_TEST.items())
def model(request):
    return request.param


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

    return klass


@pytest.fixture
def image_classification_fine_tuner_single():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = ImageClassificationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_single(input_dir)

    return klass


@pytest.fixture
def image_classification_fine_tuner_multi():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = ImageClassificationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_multi(input_dir)

    return klass


def test_vision_classification_init(image_classification_fine_tuner, model):
    model_name, labels = model
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

    assert image_classification_fine_tuner.model is not None
    assert image_classification_fine_tuner.processor is not None
    assert image_classification_fine_tuner.input is not None
    assert image_classification_fine_tuner.output is not None
    assert image_classification_fine_tuner.state is not None


def setup_test_environment_single(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)
    # Create a few sample images in different class directories
    class_names = ["class1", "class2", "class3"]
    for class_name in class_names:
        class_dir = os.path.join(test_data_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(5):  # Create 5 images per class
            img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(class_dir, f"img{i}{class_name}.png")
            img.save(img_path)


def setup_test_environment_multi(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)
    # Create a few sample images in different class directories
    class_names = ["class1", "class2", "class3"]
    for class_name in class_names:
        class_dir = os.path.join(test_data_path, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(5):  # Create 5 images per class
            img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(class_dir, f"img{i}.png")
            img.save(img_path)


def test_dataset_directory_structure(image_classification_fine_tuner_single):
    dataset_path = image_classification_fine_tuner_single.input.get()
    expected_num_classes = 3
    expected_images_per_class = 5
    classes = [
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    assert (
        len(classes) == expected_num_classes
    ), "Number of classes in dataset does not match expected."

    for class_dir in classes:
        images = os.listdir(os.path.join(dataset_path, class_dir))
        assert (
            len(images) == expected_images_per_class
        ), f"Number of images in class {class_dir} does not match expected."


def test_load_local_dataset_single_class(image_classification_fine_tuner_single, model):
    name, model_name = model
    processor_name = model_name
    model_class = "AutoModelForImageClassification"
    processor_class = "AutoProcessor"

    image_classification_fine_tuner_single.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuner_single._load_local_dataset(
        image_classification_fine_tuner_single.input.get(), is_multiclass=True
    )

    # Print or log the dataset
    print("Dataset:", dataset)

    print(dataset.features)

    for i in range(len(dataset)):
        print(f"Entry {i} label: {dataset[i]['label']}")

    assert len(dataset) == 15, f"Expected 15 images, got {len(dataset)}"
    for i in range(len(dataset)):
        assert (
            len(dataset[i]["label"]) == 1
        ), f"Expected single label for sample {i}, found {len(dataset[i]['label'])}"


def test_load_local_dataset_multi_class(image_classification_fine_tuner_multi, model):
    name, model_name = model
    processor_name = model_name
    model_class = "AutoModelForImageClassification"
    processor_class = "AutoProcessor"

    image_classification_fine_tuner_multi.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuner_multi._load_local_dataset(
        image_classification_fine_tuner_multi.input.get(), is_multiclass=True
    )

    # Print or log the dataset
    print("Dataset:", dataset)

    print(dataset.features)

    for i in range(len(dataset)):  # Adjust the range as needed
        print(f"Entry {i} label: {dataset[i]['label']}")

    assert len(dataset) == 5, f"Expected 15 images, got {len(dataset)}"
    for i in range(len(dataset)):
        assert (
            len(dataset[i]["label"]) == 3
        ), f"Expected single label for sample {i}, found {len(dataset[i]['label'])}"


def test_compute_metrics(image_classification_fine_tuner):
    # Mock predictions and labels
    num_samples = 100  # Example number of samples
    num_classes = 10  # Example number of classes
    mock_predictions = np.random.rand(num_samples, num_classes)
    mock_labels = np.random.randint(0, num_classes, num_samples)

    # Normalize predictions to mimic softmax output
    mock_predictions = mock_predictions / np.sum(
        mock_predictions, axis=1, keepdims=True
    )

    # Create a mock EvalPrediction object
    eval_pred = EvalPrediction(predictions=mock_predictions, label_ids=mock_labels)

    # Compute metrics
    computed_metrics = image_classification_fine_tuner.compute_metrics(eval_pred)

    # Calculate expected accuracy for comparison
    expected_accuracy = np.mean(np.argmax(mock_predictions, axis=1) == mock_labels)

    # Assert that the computed accuracy matches the expected accuracy
    assert np.isclose(computed_metrics["accuracy"], expected_accuracy)


def test_compute_metrics_with_arrays(image_classification_fine_tuner):
    # Mocking an EvalPrediction object
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = image_classification_fine_tuner.compute_metrics(eval_pred)

    assert "accuracy" in metrics


# Test for fine-tuning
def test_image_classfication_fine_tune(image_classification_fine_tuner, model):
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
    # kwargs = {"model_"}

    # Check that model files are created in the output directory
    assert os.path.isfile(
        os.path.join(
            image_classification_fine_tuner.output.output_folder,
            "model",
            "pytorch_model.bin",
        )
    )
    assert os.path.isfile(
        os.path.join(
            image_classification_fine_tuner.output.output_folder, "model", "config.json"
        )
    )
    assert os.path.isfile(
        os.path.join(
            image_classification_fine_tuner.output.output_folder,
            "model",
            "training_args.bin",
        )
    )
    assert os.path.isfile(
        os.path.join(
            image_classification_fine_tuner.output.output_folder,
            "model",
            "preprocessor_config.bin",
        )
    )

    del image_classification_fine_tuner.model
    del image_classification_fine_tuner.processor
