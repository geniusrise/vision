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
import shutil
import torch
from PIL import Image, ImageDraw
import pytest
import numpy as np
from transformers import TrOCRProcessor, EvalPrediction
from geniusrise import BatchInput, BatchOutput, InMemoryState
from geniusrise_vision.ocr.fine_tune import OCRFineTuner

MODELS_TO_TEST = {
    "handwritten": ("microsoft/trocr-base-handwritten", "VisionEncoderDecoderModel", "TrOCRProcessor", "cpu"),
    "printed": ("microsoft/trocr-base-printed", "VisionEncoderDecoderModel", "TrOCRProcessor", "cpu"),
}

@pytest.fixture(params=MODELS_TO_TEST.values())
def model(request):
    model_name, model_class, processor_class, device_map = request.param
    return model_name, model_class, processor_class, device_map

@pytest.fixture
def ocr_fine_tuning():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()
    state = InMemoryState()
    klass = OCRFineTuner(
        input=BatchInput(input_dir, "geniusrise-test", "test-🤗-input"),
        output=BatchOutput(output_dir, "geniusrise-test", "test-🤗-output"),
        state=state,
    )
    setup_test_environment_finetuning_ocr(input_dir)
    return klass

def create_mock_image(file_path, text):
    image = Image.new('RGB', (384, 384), color='white')
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill='black')
    image.save(file_path)

def create_mock_annotation(file_path, text):
    with open(file_path, 'w') as file:
        file.write(text)

def setup_test_environment_finetuning_ocr(base_path):
    for split in ['train', 'test']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'annotations'), exist_ok=True)

        for i in range(5):  # Creating 5 mock images and annotations per split
            img_file = f'image{i}.jpg'
            txt_file = f'image{i}.txt'
            create_mock_image(os.path.join(base_path, split, 'images', img_file), f'Sample {split} text {i}')
            create_mock_annotation(os.path.join(base_path, split, 'annotations', txt_file), f'Sample {split} text {i}')

@pytest.fixture
def mock_dataset():
    temp_dir = tempfile.mkdtemp()
    setup_test_environment_finetuning_ocr(temp_dir)
    return temp_dir

def test_vision_ocr_init(ocr_fine_tuning):
    model_name = "microsoft/trocr-base-handwritten"
    processor_name = "microsoft/trocr-base-handwritten"
    model_class = "VisionEncoderDecoderModel"
    processor_class = "TrOCRProcessor"

    ocr_fine_tuning.load_models(
        model_name=model_name,
        processor_name=processor_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
    )

    assert ocr_fine_tuning.model is not None
    assert ocr_fine_tuning.processor is not None
    assert ocr_fine_tuning.input is not None
    assert ocr_fine_tuning.output is not None
    assert ocr_fine_tuning.state is not None

def test_load_local_dataset_multi_class(ocr_fine_tuning, mock_dataset):
    ocr_fine_tuning.load_models(
        model_name="microsoft/trocr-base-handwritten",
        processor_name="microsoft/trocr-base-handwritten",
        model_class="VisionEncoderDecoderModel",
        processor_class="TrOCRProcessor",
        device_map="cpu",
    )

    train_dataset = ocr_fine_tuning._load_local_dataset(os.path.join(mock_dataset, 'train'))
    test_dataset = ocr_fine_tuning._load_local_dataset(os.path.join(mock_dataset, 'test'))

    assert len(train_dataset) == 5
    for data in train_dataset:
        assert 'pixel_values' in data and 'labels' in data

    assert len(test_dataset) == 5
    for data in test_dataset:
        assert 'pixel_values' in data and 'labels' in data

# def test_compute_metrics_for_ocr(ocr_fine_tuning):
#     pad_token_id = -100
#     pred_token_sequences = torch.tensor([[101, 102, 103, pad_token_id], [104, 105, 106, 107]])
#     true_token_sequences = torch.tensor([[101, 102, 103, pad_token_id], [104, 105, 108, 109]])

#     eval_pred = EvalPrediction(predictions=pred_token_sequences, label_ids=true_token_sequences)

#     # Compute metrics
#     metrics = ocr_fine_tuning.compute_metrics(eval_pred)

#     # Assert that the CER metric is in the computed metrics
#     assert "cer" in metrics


def test_ocr_tuning_bolt_compute_cer(ocr_fine_tuning):
    # Initialize OCRFineTuner
    model_name = "microsoft/trocr-base-handwritten"
    tokenizer_name = "microsoft/trocr-base-handwritten"
    model_class = "VisionEncoderDecoderModel"
    tokenizer_class = "TrOCRProcessor"

    ocr_fine_tuning.load_models(
        model_name=model_name,
        processor_name=tokenizer_name,
        model_class=model_class,
        processor_class=tokenizer_class,
        device_map="cpu",  # or "cuda:0" if GPU is available
    )

    # Mocking token sequences (replace with actual tokens for your model)
    # Assuming padding token ID is 1
    pad_token_id = 1
    pred_token_sequences = torch.tensor([[101, 102, 103, pad_token_id], [104, 105, 106, 107]])
    true_token_sequences = torch.tensor([[101, 102, 103, pad_token_id], [104, 105, 108, 109]])

    eval_pred = EvalPrediction(predictions=pred_token_sequences, label_ids=true_token_sequences)

    # Compute metrics
    metrics = ocr_fine_tuning.compute_metrics(eval_pred)

    # Assert that the CER metric is in the computed metrics
    assert "cer" in metrics


def test_ocr_fine_tuner_flexibility(ocr_fine_tuning, model):
    model_name, model_class, processor_class, device_map = model
    
    # Load models before fine-tuning.
    ocr_fine_tuning.load_models(
        model_name=model_name,
        processor_name=model_name,
        model_class=model_class,
        processor_class=processor_class,
        device_map=device_map,
    )

    dataset_path = mock_dataset

    dataset = ocr_fine_tuning._load_local_dataset(
        ocr_fine_tuning.input.get()
    )

    # Make sure the dataset is correctly loaded and formatted
    assert dataset is not None

    # Call the fine-tune method.
    ocr_fine_tuning.fine_tune(
        model_name=model_name,
        processor_name=model_name,
        num_train_epochs=2,  
        per_device_batch_size=3,  
        model_class=model_class,
        processor_class=processor_class,
        device_map="cpu",
        evaluate=False,
        map_data=None,
        fp16=False,  # Disable mixed precision training
    )

    # Check that model files are created in the output directory.
    output_folder = ocr_fine_tuning.output.output_folder
    assert os.path.isfile(os.path.join(output_folder, "model", "model.safetensors"))
    assert os.path.isfile(os.path.join(output_folder, "model", "training_args.bin"))
    assert os.path.isfile(os.path.join(output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(output_folder, "model", "preprocessor_config.json"))


