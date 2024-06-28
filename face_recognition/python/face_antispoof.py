import cv2
import numpy as np
from ie_module import Module

class FaceAntiSpoof(Module):
    def __init__(self, core, model_path):
        self.model = core.read_model(model=str(model_path) + '.xml', weights=str(model_path) + '.bin')
        self.exec_net = core.compile_model(model=self.model, device_name='CPU')
        
        # Debug: Print input and output layers to verify keys
        print(f"Model Inputs: {self.model.inputs}")
        print(f"Model Outputs: {self.model.outputs}")
        
        self.input_blob = next(iter(self.model.inputs))
        self.output_blob = next(iter(self.model.outputs))

    def infer(self, frame):
        # Assuming self.input_blob gives the correct input layer name
        input_blob_name = self.input_blob.get_any_name()
        
        # Get input shape
        input_shape = self.model.inputs[0].shape
        n, c, h, w = input_shape
        
        # Preprocess the frame
        blob = cv2.resize(frame, (w, h))
        blob = blob.transpose((2, 0, 1))
        blob = blob.reshape((n, c, h, w)).astype(np.float32)  # Ensure dtype is compatible
        
       
        
        # Perform inference
        results = self.exec_net.infer_new_request({input_blob_name: blob})
        # Assuming self.output_blob gives the correct output layer name
        output_blob_name = self.output_blob.get_any_name()
        return results[output_blob_name]

# Usage example
# core = SomeCoreObject()  # Initialize OpenVINO core object
# model_path = "path_to_model"
# face_anti_spoof = FaceAntiSpoof(core, model_path)
# frame = cv2.imread("path_to_image")
# result = face_anti_spoof.infer(frame)
# print(result)
