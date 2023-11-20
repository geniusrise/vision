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
from geniusrise import BatchInput, BatchOutput, State
from geniusrise.logging import setup_logger

from .bulk import ImageBulk


class VisionAPI(ImageBulk):
    """
    The VisionAPI class inherits from ImageBulk and is designed to facilitate
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

    def listen(
        self,
        model_name: str,
        model_class: str = "AutoModel",
        processor_class: str = "AutoProcessor",
        use_cuda: bool = False,
        device_map: str | Dict | None = "auto",
        max_memory={0: "24GB"},
        torchscript: bool = True,
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
            use_cuda (bool, optional): Flag to use GPU for model inference. Defaults to False.
            device_map (str | Dict | None, optional): Device mapping for model inference. Defaults to "auto".
            max_memory (Dict[int, str], optional): Maximum memory allocation for model inference. Defaults to {0: "24GB"}.
            torchscript (bool, optional): Whether to use TorchScript for model optimization. Defaults to True.
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
        self.use_cuda = use_cuda
        self.device_map = device_map
        self.max_memory = max_memory
        self.torchscript = torchscript
        self.model_args = model_args

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

        # Load the specified models with the given configurations
        self.model, self.processor = self.load_models(
            model_name=self.model_name,
            processor_name=self.processor_name,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            model_class=self.model_class,
            processor_class=self.processor_class,
            use_cuda=self.use_cuda,
            device_map=self.device_map,
            max_memory=self.max_memory,
            torchscript=self.torchscript,
            **self.model_args,
        )

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

        # Update CherryPy configuration with server details
        cherrypy.config.update(
            {
                "server.socket_host": "0.0.0.0",
                "server.socket_port": port,
                "log.screen": False,
                "tools.CORS.on": True,
            }
        )

        cherrypy.tools.CORS = cherrypy.Tool("before_handler", CORS)
        cherrypy.tree.mount(self, "/api/v1/", {"/": {"tools.CORS.on": True}})
        cherrypy.tools.CORS = cherrypy.Tool("before_finalize", CORS)
        cherrypy.engine.start()
        cherrypy.engine.block()
