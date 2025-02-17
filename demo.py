import argparse
import inspect
import os.path as osp
import sys

import cv2 as cv
import glog as log
import numpy as np

current_dir = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = osp.dirname(current_dir)
sys.path.insert(0, parent_dir)

from box import Box
from antispoof import utils_antispoof
from antispoof.demo_tools import TorchCNN, VectorCNN, FaceDetector

def pred_spoof(frame, detections, spoof_model):
    """Get prediction for all detected faces on the frame"""
    faces = []
    for rect, _ in detections:
        left, top, right, bottom = rect
        # cut face according coordinates of detections
        faces.append(frame[top:bottom, left:right])
    if faces:
        output = spoof_model.forward(faces)
        output = list(map(lambda x: x.reshape(-1), output))
        return output
    return None, None
def draw_detections(frame, detections, confidence, thresh):
    """Draws detections and labels"""
    
    for i, rect in enumerate(detections):
        left, top, right, bottom = rect[0]
        print(left, top, right, bottom)
        
        # Ensure that confidence[i] has two elements
        if len(confidence[i]) > 1:
            spoof_confidence = confidence[i][1]
        else:
            spoof_confidence = 0  # or some default value if needed

        if spoof_confidence > thresh:
            label = f'spoof: {round(spoof_confidence * 100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            real_confidence = confidence[i][0] if len(confidence[i]) > 0 else 0
            label = f'real: {round(real_confidence * 100, 3)}%'
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame


def run(params, capture, face_det, spoof_model, write_video=False):
    """Starts the anti spoofing demo"""
    fourcc = cv.VideoWriter.fourcc(*'mp4v')
    print("fourcc->>>>>>>>>>",fourcc)
    resolution = (1280,720)
    fps = 24
    writer_video = cv.VideoWriter('output_video_demo.mp4', fourcc, fps, resolution)
    win_name = 'Antispoofing Recognition'
    while cv.waitKey(1) != 27:
        has_frame, frame = capture.read()
        if not has_frame:
            return
        detections = face_det.get_detections(frame)
        confidence = pred_spoof(frame, detections, spoof_model)
        frame = draw_detections(frame, detections, confidence, params.spoof_thresh)
        cv.imshow(win_name, frame)
        if write_video:
            writer_video.write(cv.resize(frame, resolution))
    capture.release()
    writer_video.release()
    cv.destroyAllWindows()

def main():
    """Prepares data for the antispoofing recognition demo"""

    parser = argparse.ArgumentParser(description='antispoofing recognition live demo script')
    parser.add_argument('--video', type=str, default=None, help='Input video')
    parser.add_argument('--cam_id', type=int, default=0, help='Input cam')
    parser.add_argument('--config', type=str, default=r"./antispoof/configs/config.py", required=False,
                        help='Configuration file')
    parser.add_argument('--fd_model', type=str, required=False,default=r"./models/face-detection-adas-0001.xml")
    parser.add_argument('--fd_thresh', type=float, default=0.6, help='Threshold for FD')
    parser.add_argument('--spoof_thresh', type=float, default= 0.4,
                        help='Threshold for predicting spoof/real. The lower the more model oriented on spoofs')
    parser.add_argument('--spf_model', type=str, default=r"./models/anti_spoofing_480.xml",
                        help='path to .pth checkpoint of model or .xml IR OpenVINO model', required=False)
    parser.add_argument('--device', type=str, default='CPU')
    parser.add_argument('--GPU', type=int, default=0, help='specify which GPU to use')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels '
                             'impl.', type=str, default=None)
    parser.add_argument('--write_video', type=bool, default=False,
                        help='if you set this arg to True, the video of the demo will be recorded')
    args = parser.parse_args()
    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    write_video = args.write_video

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
        cap = cv.VideoCapture(args.cam_id)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*'MJPG'))
    else:
        assert args.video
        log.info('Reading from {}'.format(args.video))
        cap = cv.VideoCapture(args.video)
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc(*'MJPG'))
    assert cap.isOpened()
    face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
  
    if args.spf_model.endswith('pth.tar'):
        if not args.config:
            raise ValueError('You should pass config file to work with a Pytorch model')
        config = utils_antispoof.read_py_config(args.config)
        spoof_model = utils_antispoof.build_model(config, args, strict=True, mode='eval')
        spoof_model = TorchCNN(spoof_model, args.spf_model, config, device=device)
    else:
        assert args.spf_model.endswith('.xml')
        spoof_model = VectorCNN(args.spf_model)
    # running demo
    run(args, cap, face_detector, spoof_model, write_video)

if __name__ == '__main__':
    main()