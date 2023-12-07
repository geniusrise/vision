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
    "resnet-50": ("microsoft/resnet-50", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "vitage": ("nateraw/vit-age-classifier", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "beit": ("microsoft/beit-base-patch16-224-pt22k-ft22k", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "bit": ("google/bit-50", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "convnext": ("facebook/convnext-tiny-224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "convnextv2": ("facebook/convnextv2-tiny-1k-224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "diet": ("facebook/deit-base-distilled-patch16-224", "DeiTForImageClassification", "AutoFeatureExtractor", "cpu"),
    "diettiny": ("facebook/deit-tiny-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu"),
    "dietsmall": ("facebook/deit-small-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu"),
    "diet224": ("facebook/deit-base-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu"),
    "dinov": ("facebook/dinov2-small-imagenet1k-1-layer", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "efficientnet": ("google/efficientnet-b7", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "focalnet": ("microsoft/focalnet-tiny", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "levit128s": ("facebook/levit-128S", "LevitForImageClassification", "AutoImageProcessor", "cpu"),
    "levit128": ("facebook/levit-128", "LevitForImageClassification", "AutoImageProcessor", "cpu"),
    "levit192": ("facebook/levit-192", "LevitForImageClassification", "AutoImageProcessor", "cpu"),
    "levit256": ("facebook/levit-256", "LevitForImageClassification", "AutoImageProcessor", "cpu"),
    "levit384": ("facebook/levit-384", "LevitForImageClassification", "AutoImageProcessor", "cpu"),
    "mobilenet": ("google/mobilenet_v2_1.0_224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "mobilevit": ("apple/mobilevit-small", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "mobilevit2": ("apple/mobilevitv2-1.0-imagenet1k-256", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "poolformer": ("sail/poolformer_s12", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "pvt": ("Zetatech/pvt-tiny-224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "regnet": ("facebook/regnet-y-040", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "segformer": ("nvidia/mit-b0", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "swiftformer": ("MBZUAI/swiftformer-xs", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "swin": ("microsoft/swin-tiny-patch4-window7-224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "vit": ("google/vit-base-patch16-224", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
    "vitmsn": ("facebook/vit-msn-small", "AutoModelForImageClassification", "AutoProcessor", "cpu"),
}


@pytest.fixture(params=MODELS_TO_TEST.values())
def model(request):
    model_name, model_class, processor_class, device_map = request.param
    return model_name, model_class, processor_class, device_map


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
def image_classification_fine_tuning():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = ImageClassificationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_finetuning_single(input_dir)

    return klass


@pytest.fixture
def image_classification_fine_tuning_options():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = ImageClassificationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_finetuning_multi(input_dir)

    return klass


def setup_test_environment_finetuning_single(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)

    # Define splits and class names
    splits = ["train", "test"]
    class_names = ["class1", "class2", "class3"]

    for split in splits:
        split_dir = os.path.join(test_data_path, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Create sample images for each class
            for i in range(5):  # Create 5 images per class
                img = Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40))
                img_path = os.path.join(class_dir, f"img{i}{class_name}.png")
                img.save(img_path)


def setup_test_environment_finetuning_multi(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)

    # Define splits and class names
    splits = ["train", "test"]
    class_names = ["class1", "class2", "class3"]

    for split in splits:
        split_dir = os.path.join(test_data_path, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Create sample images for each class
            for i in range(5):  # Create 5 images per class
                img = Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40))
                img_path = os.path.join(class_dir, f"img{i}.png")
                img.save(img_path)


def test_vision_classification_init(image_classification_fine_tuner):
    model_name = "facebook/levit-128"
    processor_name = "facebook/levit-128"
    model_class = "LevitForImageClassification"
    processor_class = "AutoImageProcessor"

    image_classification_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    assert image_classification_fine_tuner.model is not None
    assert image_classification_fine_tuner.processor is not None
    assert image_classification_fine_tuner.input is not None
    assert image_classification_fine_tuner.output is not None
    assert image_classification_fine_tuner.state is not None


def test_dataset_directory_structure(image_classification_fine_tuning):
    dataset_path = image_classification_fine_tuning.input.get()
    expected_num_classes = 3
    expected_images_per_class = 5
    splits = ["train", "test"]  # Adjust if your dataset has different splits

    for split in splits:
        split_dir = os.path.join(dataset_path, split)
        classes = [
            d
            for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ]
        assert len(classes) == expected_num_classes, f"Number of classes in {split} split does not match expected."

        # Optionally, check the number of images in each class
        for class_dir in classes:
            class_path = os.path.join(split_dir, class_dir)
            images = [
                img
                for img in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, img))
            ]
            assert len(images) == expected_images_per_class, f"Number of images in class {class_dir} in {split} split does not match expected."


def test_load_local_dataset_single_class(image_classification_fine_tuning):
    model_name = "facebook/levit-128"
    processor_name = "facebook/levit-128"
    model_class = "LevitForImageClassification"
    processor_class = "AutoImageProcessor"


    image_classification_fine_tuning.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuning._load_local_dataset(
        image_classification_fine_tuning.input.get(), is_multiclass=False
    )

    assert len(dataset) == 15
    

def test_load_local_dataset_multi_class(image_classification_fine_tuning_options):
    model_name = "facebook/levit-128"
    processor_name = "facebook/levit-128"
    model_class = "LevitForImageClassification"
    processor_class = "AutoImageProcessor"


    image_classification_fine_tuning_options.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuning_options._load_local_dataset(
        image_classification_fine_tuning_options.input.get(), is_multiclass=True
    )

    # Iterate and print the contents
    for i, data in enumerate(dataset):
        print(f"Sample {i}:")
        for tensor in data:
            print(tensor)
        print()

    assert len(dataset) == 5


def test_compute_metrics_with_arrays(image_classification_fine_tuner):
    # Mocking an EvalPrediction object
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = image_classification_fine_tuner.compute_metrics(eval_pred)

    assert "accuracy" in metrics


def test_image_classification_fine_tune(image_classification_fine_tuning_options):
    # Ensure the model class and processor class are set correctly.
    model_name = "facebook/levit-128"
    processor_name = "facebook/levit-128"
    model_class = "LevitForImageClassification"
    processor_class = "AutoImageProcessor"

    # Load models before fine-tuning.
    image_classification_fine_tuning_options.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = image_classification_fine_tuning_options._load_local_dataset(
        image_classification_fine_tuning_options.input.get(), is_multiclass=False
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    image_classification_fine_tuning_options.fine_tune(
        model_name=model_name,
        processor_name=processor_name,
        num_train_epochs=2,
        per_device_batch_size=3,
        model_class=model_class,
        evaluate=False,
        processor_class=processor_class,
        device_map="cpu",
        dataset=dataset,
    )

    # Check that model files are created in the output directory.
    output_folder = image_classification_fine_tuning_options.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(output_folder, "model", "preprocessor_config.json"))


def test_model_flexibility(image_classification_fine_tuning_options, model):
    model_name, model_class, processor_class, device_map = model
    
    # Load models before fine-tuning.
    image_classification_fine_tuning_options.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map=device_map,
    )

    dataset = image_classification_fine_tuning_options._load_local_dataset(
        image_classification_fine_tuning_options.input.get(), is_multiclass=False
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    image_classification_fine_tuning_options.fine_tune(
        model_name=model_name,
        processor_name=model_name,
        num_train_epochs=2,
        per_device_batch_size=3,
        model_class=model_class,
        evaluate=False,
        processor_class=processor_class,
        device_map="cpu",
        dataset=dataset,
    )

    # Check that model files are created in the output directory.
    output_folder = image_classification_fine_tuning_options.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(output_folder, "model", "preprocessor_config.json"))
    


