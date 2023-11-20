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

import numpy as np
import pytest
import torch
from geniusrise.core import BatchInput, BatchOutput, InMemoryState
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from transformers import EvalPrediction

from geniusrise_vision.base.fine_tune import VisionFineTuner


class DictDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = torch.tensor(label, dtype=torch.long)
        return {"pixel_values": image, "labels": label}


class TestVisionFineTuner(VisionFineTuner):
    def load_dataset(self, dataset_path, **kwargs):
        # Define the transforms
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel RGB
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),  # Normalize for 3 channels
            ]
        )
        # Load the MNIST dataset with the specified transforms
        mnist_dataset = MNIST(root=dataset_path, train=True, download=True, transform=transform)

        # Wrap the MNIST dataset in the DictDataset wrapper
        dataset = DictDataset(mnist_dataset)

        # Return the wrapped dataset
        return dataset


@pytest.fixture
def bolt():
    input_dir = tempfile.mkdtemp()
    output_dir = tempfile.mkdtemp()

    input = BatchInput(input_dir, "geniusrise-test", "test-🤗-input")
    output = BatchOutput(output_dir, "geniusrise-test", "test-🤗-output")
    state = InMemoryState()

    return TestVisionFineTuner(
        input=input,
        output=output,
        state=state,
        evaluate=False,
        batch_size=32,
    )


def test_bolt_init(bolt):
    assert bolt.input is not None
    assert bolt.output is not None
    assert bolt.state is not None


def test_load_dataset(bolt):
    bolt.model_name = "microsoft/resnet-50"
    bolt.processor_name = "microsoft/resnet-50"
    bolt.model_class = "AutoModel"
    bolt.processor_class = "AutoProcessor"
    bolt.load_models(
        model_name=bolt.model_name,
        processor_name=bolt.processor_name,
        model_class=bolt.model_class,
        processor_class=bolt.processor_class,
        device_map=None,
    )
    dataset = bolt.load_dataset("fake_path")
    assert dataset is not None
    assert len(dataset) >= 100

    del bolt.model
    del bolt.processor
    torch.cuda.empty_cache()


def test_fine_tune(bolt):
    bolt.fine_tune(
        model_name="google/vit-base-patch16-224",
        processor_name="google/vit-base-patch16-224",
        num_train_epochs=1,
        per_device_batch_size=2,
        model_class="AutoModelForImageClassification",
        processor_class="AutoProcessor",
        evaluate=False,
        device_map="auto",
    )
    bolt.upload_to_hf_hub(
        hf_repo_id="ixaxaar/geniusrise-hf-base-test-repo",
        hf_commit_message="testing base fine tuner",
        hf_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
        hf_private=False,
        hf_create_pr=True,
    )

    # Check that model files are created in the output directory
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "config.json"))
    assert os.path.isfile(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))

    del bolt.model
    del bolt.processor
    torch.cuda.empty_cache()


def test_compute_metrics(bolt):
    # Mocking an EvalPrediction object
    logits = np.array([[0.6, 0.4], [0.4, 0.6]])
    labels = np.array([0, 1])
    eval_pred = EvalPrediction(predictions=logits, label_ids=labels)

    metrics = bolt.compute_metrics(eval_pred)

    assert "accuracy" in metrics


models = {
    "small": "google/mobilenet_v2_1.0_224",
    "medium": "google/vit-base-patch16-224",
    "large": "microsoft/resnet-50",
}


@pytest.mark.parametrize(
    "model_size, use_accelerate",
    [
        # small
        ("small", False),
        ("small", True),
        # medium
        ("medium", False),
        ("medium", True),
        # large
        ("large", False),
        ("large", True),
    ],
)
def test_fine_tune_options(bolt, model_size, use_accelerate):
    model = models[model_size]

    bolt.fine_tune(
        model_name=model,
        processor_name=model,
        model_class="AutoModelForImageClassification",
        processor_class="AutoProcessor",
        num_train_epochs=1,
        per_device_batch_size=2,
        device_map=None,
    )

    # Verify the model has been fine-tuned by checking the existence of model files
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "config.json"))
    assert os.path.exists(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))

    # Clear the output directory for the next test
    try:
        os.remove(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))
    except FileNotFoundError:
        pass

    del bolt.model
    del bolt.processor
    torch.cuda.empty_cache()

    # Clear the output directory for the next test
    try:
        os.remove(os.path.join(bolt.output.output_folder, "model", "pytorch_model.bin"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))
    except Exception as _:
        pass

    del bolt.model
    del bolt.processor
    torch.cuda.empty_cache()
