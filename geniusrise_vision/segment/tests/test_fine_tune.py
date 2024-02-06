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
import json
from geniusrise import BatchInput, BatchOutput, InMemoryState
from PIL import Image
from geniusrise_vision.segment.fine_tune import SegmentationFineTuner 

MODELS_TO_TEST = {
    #"beit": ("microsoft/beit-base-finetuned-ade-640-640", "AutoModelForSemanticSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    # "data2vec": ("facebook/data2vec-vision-base","Data2VecVisionForSemanticSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    #"dpt": ("Intel/dpt-large-ade","AutoModelForSemanticSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    "mobilenetv2": ("google/deeplabv3_mobilenet_v2_1.0_513", "AutoModelForSemanticSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    "mobilevit": ("apple/deeplabv3-mobilevit-small", "AutoModelForSemanticSegmentation", "MobileViTImageProcessor", "cpu", "semantic"),
    "mobilevit2": ("apple/mobilevitv2-1.0-imagenet1k-256", "AutoModelForSemanticSegmentation", "MobileViTImageProcessor", "cpu", "semantic"),
    "segformer": ("nvidia/segformer-b0-finetuned-ade-512-512",  "AutoModelForSemanticSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    "mask2formersemantic": ("facebook/mask2former-swin-small-ade-semantic", "Mask2FormerForUniversalSegmentation", "AutoImageProcessor", "cpu", "semantic"),
    "maskformersemantic": ("facebook/maskformer-swin-base-ade", "MaskFormerForInstanceSegmentation", "AutoImageProcessor", "cpu", "semantic"),
}


@pytest.fixture(params=MODELS_TO_TEST.values())
def model(request):
    model_name, model_class, processor_class, device_map, sub_task  = request.param
    return model_name, model_class, processor_class, device_map, sub_task


@pytest.fixture
def segment_fine_tuner():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = SegmentationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_segmentation(input_dir)

    return klass


@pytest.fixture
def segment_maskformer_fine_tuner():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = SegmentationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_environment_maskformer(input_dir)

    return klass


@pytest.fixture
def segment_maskformer_test_folder_fine_tuner():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = SegmentationFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )

    # Create a small dataset
    setup_test_folder_maskformer(input_dir)

    return klass


def setup_test_environment_segmentation(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)

    # Define splits
    splits = ["train", "test"]

    for split in splits:
        split_dir = os.path.join(test_data_path, split)
        os.makedirs(split_dir, exist_ok=True)

        # Create sample images and segmentation masks
        for i in range(5):  # Create 5 images and masks
            img = Image.new("RGB", (224, 224), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(split_dir, f"image{i}.jpg")
            img.save(img_path)

            # Create a corresponding segmentation map
            # Here, we just create a dummy map with all pixels labeled as 'i'
            seg_map = Image.new("L", (224, 224), color=i)  # 'L' mode for grayscale
            seg_map_path = os.path.join(split_dir, f"image{i}_seg.png")
            seg_map.save(seg_map_path)

            # Create JSON metadata files
            json_path = os.path.join(split_dir, f"image{i}.json")
            with open(json_path, 'w') as json_file:
                 json.dump({'some_key': 'some_value'}, json_file)


def setup_test_environment_maskformer(test_data_path):
    os.makedirs(test_data_path, exist_ok=True)

    # Define splits
    splits = ["train", "test"]

    for split in splits:
        split_dir = os.path.join(test_data_path, split)
        os.makedirs(split_dir, exist_ok=True)

        # Create sample images and segmentation masks
        for i in range(5):  # Create 5 images and masks
            img = Image.new("RGB", (512, 512), color=(i * 40, i * 40, i * 40))
            img_path = os.path.join(split_dir, f"image{i}.jpg")
            img.save(img_path)

            # Create a corresponding segmentation map
            # Here, we just create a dummy map with all pixels labeled as 'i'
            seg_map = Image.new("L", (512, 512), color=i)  # 'L' mode for grayscale
            seg_map_path = os.path.join(split_dir, f"image{i}_seg.png")
            seg_map.save(seg_map_path)

            # Create JSON metadata files with inst2class
            json_path = os.path.join(split_dir, f"image{i}.json")
            metadata = {
                'segments_info': [
                    {
                        'id': i,
                        'category_id': i,  # Assuming each segment has a unique class
                    }
                ]
            }
            with open(json_path, 'w') as json_file:
                json.dump(metadata, json_file)

IMAGE_FOLDER = 'test_images_segment_finetune'

def setup_test_folder_maskformer(test_data_path):
    # Create the test data directory if it doesn't exist
    os.makedirs(test_data_path, exist_ok=True)

    # Define splits
    splits = ["train", "test"]

    for split in splits:
        split_dir = os.path.join(test_data_path, split)
        os.makedirs(split_dir, exist_ok=True)

        # Iterate over all files in the IMAGE_FOLDER
        for file_name in os.listdir(IMAGE_FOLDER):
            file_path = os.path.join(IMAGE_FOLDER, file_name)

            # Check if the file is an image
            if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Open the image
                with Image.open(file_path) as img:
                    # Construct the path for saving the image
                    destination_path = os.path.join(split_dir, file_name)
                    # Save the image to the new path
                    img.save(destination_path)


def test_vision_segmentation_init(segment_fine_tuner):
    model_name = "facebook/data2vec-vision-base"
    processor_name = "facebook/data2vec-vision-base"
    model_class = "AutoModelForSemanticSegmentation"
    processor_class = "AutoImageProcessor"

    segment_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    assert segment_fine_tuner.model is not None
    assert segment_fine_tuner.processor is not None
    assert segment_fine_tuner.input is not None
    assert segment_fine_tuner.output is not None
    assert segment_fine_tuner.state is not None


def test_dataset_directory_structure(segment_fine_tuner):
    dataset_path = segment_fine_tuner.input.get()
    expected_images_per_split = 5
    splits = ["train", "test"]  # Adjust if your dataset has different splits

    for split in splits:
        split_dir = os.path.join(dataset_path, split)

        # Get the list of images and segmentation maps
        images = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
        seg_maps = [f for f in os.listdir(split_dir) if f.endswith('_seg.png')]

        # Check if the number of images and segmentation maps matches the expected count
        assert len(images) == expected_images_per_split, f"Number of images in {split} split does not match expected."
        assert len(seg_maps) == expected_images_per_split, f"Number of segmentation maps in {split} split does not match expected."

        # Check if each image has a corresponding segmentation map
        for img_file in images:
            base_name = os.path.splitext(img_file)[0]
            seg_map_file = f"{base_name}_seg.png"
            assert seg_map_file in seg_maps, f"Segmentation map for image {img_file} not found in {split} split."


def test_load_local_dataset(segment_fine_tuner):
    model_name = "facebook/data2vec-vision-base"
    processor_name = "facebook/data2vec-vision-base"
    model_class = "AutoModelForSemanticSegmentation"
    processor_class = "AutoImageProcessor"


    segment_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    dataset = segment_fine_tuner._load_local_dataset(
        segment_fine_tuner.input.get()
    )

    assert len(dataset) == 10


def test_compute_metrics_with_arrays(segment_fine_tuner):
    model_name = "facebook/data2vec-vision-base"
    processor_name = "facebook/data2vec-vision-base"
    model_class = "AutoModelForSemanticSegmentation"
    processor_class = "AutoImageProcessor"


    segment_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    # Assuming 2 classes, 2 samples, and a 4x4 image size for simplicity
    num_classes = 2
    height, width = 4, 4
    batch_size = 2

    # Create mock logits as a 4D tensor and labels as a 3D tensor
    logits = np.random.rand(batch_size, num_classes, height, width)
    labels = np.random.randint(0, num_classes, (batch_size, height, width))

    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = segment_fine_tuner.compute_metrics(eval_pred)

    # Adjust the assert statement based on actual metrics computed
    assert "per_category_accuracy" in metrics, "Expected 'per_category_accuracy' in computed metrics"
    assert "per_category_iou" in metrics, "Expected 'per_category_iou' in computed metrics"


def test_model_flexibility(segment_fine_tuner, model):
    model_name, model_class, processor_class, device_map, subtask = model
    
    # Load models before fine-tuning.
    segment_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map=device_map,
    )

    dataset = segment_fine_tuner._load_local_dataset(
        segment_fine_tuner.input.get()
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    segment_fine_tuner.fine_tune(
        model_name=model_name,
        processor_name=model_name,
        num_train_epochs=2,
        per_device_batch_size=3,
        model_class=model_class,
        evaluate=False,
        processor_class=processor_class,
        device_map="cpu",
        dataset=dataset,
        subtask=subtask, 
    )

    # Check that model files are created in the output directory.
    output_folder = segment_fine_tuner.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(output_folder, "model", "preprocessor_config.json"))

def test_model_segment(segment_maskformer_fine_tuner, model):
    model_name, model_class, processor_class, device_map, subtask = model

    # Load models before fine-tuning.
    segment_maskformer_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map=device_map,
    )

    
    dataset = segment_maskformer_fine_tuner._load_local_dataset(
        segment_maskformer_fine_tuner.input.get()
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    segment_maskformer_fine_tuner.fine_tune(
        model_name=model_name,
        processor_name=model_name,
        num_train_epochs=2,
        per_device_batch_size=3,
        model_class=model_class,
        evaluate=False,
        processor_class=processor_class,
        device_map="cpu",
        dataset=dataset,
        subtask=subtask,
    )

    # Check that model files are created in the output directory.
    output_folder = segment_maskformer_fine_tuner.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))
    # assert os.path.isfile(os.path.join(output_folder, "model", "preprocessor_config.json"))


def test_model_segment_folder(segment_maskformer_test_folder_fine_tuner, model):
    model_name, model_class, processor_class, device_map, subtask = model

    # Load models before fine-tuning.
    segment_maskformer_test_folder_fine_tuner.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map=device_map,
    )

    
    dataset = segment_maskformer_test_folder_fine_tuner._load_local_dataset(
        segment_maskformer_test_folder_fine_tuner.input.get()
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    segment_maskformer_test_folder_fine_tuner.fine_tune(
        model_name=model_name,
        processor_name=model_name,
        num_train_epochs=2,
        per_device_batch_size=3,
        model_class=model_class,
        evaluate=False,
        processor_class=processor_class,
        device_map="cpu",
        dataset=dataset,
        subtask=subtask,
    )

    # Check that model files are created in the output directory.
    output_folder = segment_maskformer_test_folder_fine_tuner.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))