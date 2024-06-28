"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
import os

import glog as log
import numpy as np
import cv2 as cv

from openvino.runtime import Core


class IEModel:
    """Class for inference of models in the Inference Engine format"""
    def __init__(self, exec_net, inputs, input_key, output_key, switch_rb=False):
        self.net = exec_net
        self.inputs = inputs
        self.input_key_name = input_key.get_any_name()  # Sử dụng tên của đầu vào
        self.output_key_name = output_key.get_any_name()  # Sử dụng tên của đầu ra
        self.reqs_ids = []
        self.switch_rb = switch_rb

    def _preprocess(self, img):
        _, _, h, w = self.get_input_shape()
        if self.switch_rb:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.expand_dims(cv.resize(img, (w, h)).transpose(2, 0, 1), axis=0)
        return img

    def forward(self, img):
        """Performs forward pass of the wrapped IE model"""
        res = self.net.infer_new_request(inputs={self.input_key_name: self._preprocess(img)})
        return np.copy(res[self.output_key_name])

    def forward_async(self, img):
        id = len(self.reqs_ids)
        self.net.start_async(request_id=id, inputs={self.input_key_name: self._preprocess(img)})
        self.reqs_ids.append(id)

    def grab_all_async(self):
        outputs = []
        for id in self.reqs_ids:
            self.net.requests[id].wait()
            res = self.net.requests[id].output_blobs[self.output_key_name].buffer
            outputs.append(np.copy(res))
        self.reqs_ids = []
        return outputs

    def get_input_shape(self):
        for output in self.inputs:
            return output.shape
        # return self.inputs[self.input_key_name].tensor_desc.dims


def load_ie_model(model_xml, device, plugin_dir, cpu_extension='', num_reqs=1, **kwargs):
    """Loads a model in the Inference Engine format"""
    # Plugin initialization for specified device and load extensions library if specified
    log.info(f"Initializing Inference Engine plugin for {device}")

    core = Core()
    if cpu_extension and 'CPU' in device:
        core.add_extension(cpu_extension, 'CPU')

    # Read IR
    log.info("Loading network")
    net = core.read_model(model=model_xml, weights=os.path.splitext(model_xml)[0] + ".bin")

    assert len(net.inputs) == 1 or len(net.inputs) == 2, \
        "Supports topologies with only 1 or 2 inputs"
    assert len(net.outputs) == 1 or len(net.outputs) == 4 or 5, \
        "Supports topologies with only 1, 4 hoặc 5 outputs"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = core.compile_model(model=net, device_name=device)

    # Set number of requests
    exec_net.requests = [exec_net.create_infer_request() for _ in range(num_reqs)]
    
    model = IEModel(exec_net, net.inputs, input_blob, out_blob, **kwargs)
    return model