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

from typing import Any, Dict, Optional

import cherrypy
import threading
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger
import json

from .bulk import VisionBulk


# Define a global lock for sequential access control
sequential_lock = threading.Lock()


class VisionAPI(VisionBulk):
    """
    The VisionAPI class inherits from VisionBulk and is designed to facilitate
    the handling of vision-based tasks using a pre-trained machine learning model.
    It sets up a server to process image-related requests using a specified model.
    """

    model: Any
    processor: Any

    def __init__(
        self,
        input: BatchInput,
        output: BatchOutput,
        state: State,
    ):
        """
        Initializes the VisionAPI object with batch input, output, and state.

        Args:
            input (BatchInput): Object to handle batch input operations.
            output (BatchOutput): Object to handle batch output operations.
            state (State): Object to maintain the state of the API.
        """
        super().__init__(input=input, output=output, state=state)
        self.log = setup_logger(self)

    def validate_password(self, realm, username, password):
        """
        Validate the username and password against expected values.

        Args:
            realm (str): The authentication realm.
            username (str): The provided username.
            password (str): The provided password.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        return username == self.username and password == self.password

    def listen(
        self,
        model_name: str,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        use_cuda: bool = False,
        precision: str = "float16",
        quantization: int = 0,
        torchscript: bool = False,
        compile: bool = False,
        flash_attention: bool = False,
        better_transformers: bool = False,
        concurrent_queries: bool = False,
        endpoint: str = "*",
        port: int = 3000,
        cors_domain: str = "http://localhost:3000",
        username: Optional[str] = None,
        password: Optional[str] = None,
        **model_args: Any,
    ) -> None:
        """
        Configures and starts a CherryPy server to listen for image processing requests.

        Args:
            model_name (str): The name of the pre-trained vision model.
            model_class (str, optional): The class of the pre-trained vision model. Defaults to "AutoModel".
            processor_class (str, optional): The class of the processor for input image preprocessing. Defaults to "AutoProcessor".
            device_map (str | Dict | None, optional): Device mapping for model inference. Defaults to "auto".
            max_memory (Dict[int, str], optional): Maximum memory allocation for model inference. Defaults to {0: "24GB"}.
            precision (str): The floating-point precision to be used by the model. Options are 'float32', 'float16', 'bfloat16'.
            quantization (int): The bit level for model quantization (0 for none, 8 for 8-bit quantization).
            torchscript (bool, optional): Whether to use TorchScript for model optimization. Defaults to True.
            compile (bool, optional): Whether to compile the model before fine-tuning. Defaults to False.
            flash_attention (bool): Whether to use flash attention 2. Default is False.
            better_transformers (bool): Flag to enable Better Transformers optimization for faster processing.
            concurrent_queries: (bool): Whether the API supports concurrent API calls (usually false).
            endpoint (str, optional): The network endpoint for the server. Defaults to "*".
            port (int, optional): The network port for the server. Defaults to 3000.
            cors_domain (str, optional): The domain to allow for CORS requests. Defaults to "http://localhost:3000".
            username (Optional[str], optional): Username for server authentication. Defaults to None.
            password (Optional[str], optional): Password for server authentication. Defaults to None.
            **model_args (Any): Additional arguments for the vision model.
        """
        self.model_name = model_name
        self.model_class = model_class
        self.processor_class = processor_class
        self.device_map = device_map
        self.max_memory = max_memory
        self.use_cuda = use_cuda
        self.precision = precision
        self.quantization = quantization
        self.torchscript = torchscript
        self.compile = compile
        self.flash_attention = flash_attention
        self.better_transformers = better_transformers
        self.concurrent_queries = concurrent_queries
        self.model_args = model_args
        self.username = username
        self.password = password

        # Extract model revision details if specified in model_name
        if ":" in model_name:
            model_revision = model_name.split(":")[1]
            processor_revision = model_name.split(":")[1]
            model_name = model_name.split(":")[0]
            processor_name = model_name
        else:
            model_revision = None
            processor_revision = None

        # Store model and processor details
        self.model_name = model_name
        self.model_revision = model_revision
        self.processor_name = model_name
        self.processor_revision = processor_revision

        if model_name not in ["easyocr", "mmocr", "paddleocr"]:
            # Load the specified models with the given configurations
            self.model, self.processor = self.load_models(
                model_name=self.model_name,
                processor_name=self.processor_name,
                model_revision=self.model_revision,
                processor_revision=self.processor_revision,
                model_class=self.model_class,
                processor_class=self.processor_class,
                use_cuda=self.use_cuda,
                precision=self.precision,
                quantization=self.quantization,
                device_map=self.device_map,
                max_memory=self.max_memory,
                torchscript=self.torchscript,
                compile=self.compile,
                flash_attention=self.flash_attention,
                better_transformers=self.better_transformers,
                # **self.model_args,
            )

        def sequential_locker():
            if self.concurrent_queries:
                sequential_lock.acquire()

        def sequential_unlocker():
            if self.concurrent_queries:
                sequential_lock.release()

        def CORS():
            """
            Configures Cross-Origin Resource Sharing (CORS) for the server.
            This allows the server to accept requests from the specified domain.
            """
            # Setting up CORS headers
            cherrypy.response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
            cherrypy.response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            cherrypy.response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            cherrypy.response.headers["Access-Control-Allow-Credentials"] = "true"

            if cherrypy.request.method == "OPTIONS":
                cherrypy.response.status = 200
                return True

        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
                "tools.CORS.on": True,
                "error_page.400": error_page,
                "error_page.401": error_page,
                "error_page.402": error_page,
                "error_page.403": error_page,
                "error_page.404": error_page,
                "error_page.405": error_page,
                "error_page.406": error_page,
                "error_page.408": error_page,
                "error_page.415": error_page,
                "error_page.429": error_page,
                "error_page.500": error_page,
                "error_page.501": error_page,
                "error_page.502": error_page,
                "error_page.503": error_page,
                "error_page.504": error_page,
                "error_page.default": error_page,
            }
        )

        if username and password:
            # Configure basic authentication
            conf = {
                "/": {
                    "tools.sequential_locker.on": True,
                    "tools.sequential_unlocker.on": True,
                    "tools.auth_basic.on": True,
                    "tools.auth_basic.realm": "geniusrise",
                    "tools.auth_basic.checkpassword": self.validate_password,
                    "tools.CORS.on": True,
                }
            }
        else:
            # Configuration without authentication
            conf = {
                "/": {
                    "tools.sequential_locker.on": True,
                    "tools.sequential_unlocker.on": True,
                    "tools.CORS.on": True,
                }
            }

        cherrypy.tools.sequential_locker = cherrypy.Tool("before_handler", sequential_locker)
        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", conf)
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.tools.sequential_unlocker = cherrypy.Tool("before_finalize", sequential_unlocker)
        cherrypy.engine.start()
        cherrypy.engine.block()


def error_page(status, message, traceback, version):
    response = {
        "status": status,
        "message": message,
    }
    return json.dumps(response)
