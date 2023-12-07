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
from torchvision import transforms, datasets
from transformers import EvalPrediction
from PIL import Image
from geniusrise_vision.base.fine_tune import VisionFineTuner


class TestVisionFineTuner(VisionFineTuner):
    def load_dataset(self, dataset_path, **kwargs):
        os.makedirs(dataset_path, exist_ok=True)

        # Define splits and class names
        splits = ["train", "test"]
        class_names = ["class1", "class2", "class3"]

        for split in splits:
            split_dir = os.path.join(dataset_path, split)
            os.makedirs(split_dir, exist_ok=True)

            for class_name in class_names:
                class_dir = os.path.join(split_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Create sample images for each class
                for i in range(5):  # Create 5 images per class
                    img = Image.new("RGB", (100, 100), color=(i * 40, i * 40, i * 40))
                    img_path = os.path.join(class_dir, f"img{i}{class_name}.png")
                    img.save(img_path)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor()        
        ])

        # Load the dataset using ImageFolder
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

        return dataset

    @staticmethod
    def custom_vision_collator(batch):
        pixel_values = torch.stack([item[0] for item in batch])

        # Check if each item in the batch has more than one element (i.e., includes a label)
        labels = torch.tensor([item[1] for item in batch]) if all(len(item) > 1 for item in batch) else None

        return {"pixel_values": pixel_values, "labels": labels}


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
        device_map="cpu",
    )
    dataset = bolt.load_dataset("fake_path")
    assert dataset is not None
    assert len(dataset) >= 20

    del bolt.model
    del bolt.processor
    torch.cuda.empty_cache()


def test_fine_tune(bolt):
    bolt.fine_tune(
        model_name = "facebook/levit-128",
        processor_name = "facebook/levit-128",
        model_class = "LevitForImageClassification",
        processor_class = "AutoImageProcessor",
        per_device_batch_size=2,
        num_train_epochs=1,
        evaluate=False,
        device_map="cpu",
    )
    bolt.upload_to_hf_hub(
        hf_repo_id="ixaxaar/geniusrise-hf-base-test-repo",
        hf_commit_message="testing base fine tuner",
        hf_token=os.getenv("HUGGINGFACE_ACCESS_TOKEN"),
        hf_private=False,
        hf_create_pr=True,
    )

    # Check that model files are created in the output directory
    assert os.path.isfile(
        os.path.join(bolt.output.output_folder, "model", "model.safetensors")
    )
    assert os.path.isfile(
        os.path.join(bolt.output.output_folder, "model", "config.json")
    )
    assert os.path.isfile(
        os.path.join(bolt.output.output_folder, "model", "training_args.bin")
    )
    assert os.path.isfile(
        os.path.join(bolt.output.output_folder, "model", "preprocessor_config.json")
    )

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


# models = {
#     "resnet-50": "microsoft/resnet-50",
#     "vitage": "nateraw/vit-age-classifier",
#     "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
#     "bit": "google/bit-50",
#     "convnext": "facebook/convnext-tiny-224",
#     "convnextv2": "facebook/convnextv2-tiny-1k-224",
#     "diet": "facebook/deit-base-distilled-patch16-224",
#     "dinov": "facebook/dinov2-small-imagenet1k-1-layer",
#     "efficientnet": "google/efficientnet-b7",
#     "focalnet": "microsoft/focalnet-tiny",
#     "levit128s": "facebook/levit-128S",
#     "levit128":"facebook/levit-128", 
#     "levit192": "facebook/levit-192", 
#     "levit256": "facebook/levit-256", 
#     "levit384": "facebook/levit-384",
#     "mobilenet": "google/mobilenet_v2_1.0_224",
#     "mobilevit": "apple/mobilevit-small",
#     "mobilevit2": "apple/mobilevitv2-1.0-imagenet1k-256",
#     "poolformer": "sail/poolformer_s12",
#     "pvt": "Zetatech/pvt-tiny-224",
#     "regnet": "facebook/regnet-y-040",
#     "segformer": "nvidia/mit-b0",
#     "swiftformer": "MBZUAI/swiftformer-xs",
#     "swin": "microsoft/swin-tiny-patch4-window7-224",
#     "vit": "google/vit-base-patch16-224",
#     "vithybrid": "google/vit-hybrid-base-bit-384",
#     "vitmsn": "facebook/vit-msn-small",
# }

models = {
    "resnet-50": [("microsoft/resnet-50", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "vitage": [("nateraw/vit-age-classifier", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "beit": [("microsoft/beit-base-patch16-224-pt22k-ft22k", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "bit": [("google/bit-50", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "convnext": [("facebook/convnext-tiny-224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "convnextv2": [("facebook/convnextv2-tiny-1k-224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "diet": [("facebook/deit-base-distilled-patch16-224", "DeiTForImageClassification", "AutoFeatureExtractor", "cpu")],
    "diettiny": [("facebook/deit-tiny-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu")],
    "dietsmall": [("facebook/deit-small-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu")],
    "diet224": [("facebook/deit-base-patch16-224", "AutoModelForImageClassification", "AutoFeatureExtractor", "cpu")],
    "dinov": [("facebook/dinov2-small-imagenet1k-1-layer", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "efficientnet": [("google/efficientnet-b7", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "focalnet": [("microsoft/focalnet-tiny", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "levit128s": [("facebook/levit-128S", "LevitForImageClassification", "AutoImageProcessor", "cpu")],
    "levit128": [("facebook/levit-128", "LevitForImageClassification", "AutoImageProcessor", "cpu")],
    "levit192": [("facebook/levit-192", "LevitForImageClassification", "AutoImageProcessor", "cpu")],
    "levit256": [("facebook/levit-256", "LevitForImageClassification", "AutoImageProcessor", "cpu")],
    "levit384": [("facebook/levit-384", "LevitForImageClassification", "AutoImageProcessor", "cpu")],
    "mobilenet": [("google/mobilenet_v2_1.0_224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "mobilevit": [("apple/mobilevit-small", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "mobilevit2": [("apple/mobilevitv2-1.0-imagenet1k-256", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "poolformer": [("sail/poolformer_s12", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "pvt": [("Zetatech/pvt-tiny-224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "regnet": [("facebook/regnet-y-040", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "segformer": [("nvidia/mit-b0", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "swiftformer": [("MBZUAI/swiftformer-xs", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "swin": [("microsoft/swin-tiny-patch4-window7-224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "vit": [("google/vit-base-patch16-224", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
    "vitmsn": [("facebook/vit-msn-small", "AutoModelForImageClassification", "AutoProcessor", "cpu")],
}

# Flatten the models dictionary for parametrization
model_configurations = [(key, *config) for key, configs in models.items() for config in configs]

# Parametrize the test over the model configurations
@pytest.mark.parametrize("model_key,model_name,model_class,processor_class,device_map", model_configurations)
def test_fine_tune_options(bolt, model_key, model_name, model_class, processor_class, device_map):
    bolt.fine_tune(
        model_name=model_name,
        processor_name=model_name,  # Assuming model_name and processor_name are the same
        model_class=model_class,
        processor_class=processor_class,
        num_train_epochs=1,
        per_device_batch_size=2,
        device_map=device_map,
    )

    # Verify the model has been fine-tuned by checking the existence of model files
    assert os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "model.safetensors")
    )
    assert os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "config.json")
    )
    assert os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "training_args.bin")
    )
    assert os.path.exists(
        os.path.join(bolt.output.output_folder, "model", "preprocessor_config.json")
    )

    # Clear the output directory for the next test
    try:
        os.remove(os.path.join(bolt.output.output_folder, "model", "model.safetensors"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))
        os.remove(
            os.path.join(bolt.output.output_folder, "model", "preprocessor_config.json")
        )
    except FileNotFoundError:
        pass

    torch.cuda.empty_cache()

    # Clear the output directory for the next test
    try:
        os.remove(os.path.join(bolt.output.output_folder, "model", "model.safetensors"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "config.json"))
        os.remove(os.path.join(bolt.output.output_folder, "model", "training_args.bin"))
        os.remove(
            os.path.join(bolt.output.output_folder, "model", "preprocessor_config.json")
        )
    except Exception as _:
        pass

    torch.cuda.empty_cache()
