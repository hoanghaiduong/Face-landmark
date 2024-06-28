import os
import time
import cv2
import sys
import numpy as np
from pathlib import Path

# Thêm đường dẫn tới thư mục chứa các module nhận diện khuôn mặt
sys.path.append(str(Path(__file__).resolve().parents[0] / "face_recognition/python"))
sys.path.append(str(Path(__file__).resolve().parents[0] / "common/python"))
sys.path.append(str(Path(__file__).resolve().parents[0] / "common/python/model_zoo"))
from argparse import ArgumentParser
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_socketio import SocketIO, emit
from openvino import Core
from time import perf_counter
import logging as log
from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from face_antispoof import FaceAntiSpoof
from utils import crop
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from helpers import resolution

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

DEVICE_KINDS = ["CPU", "GPU", "HETERO"]
FRAME_PROCESSOR = None
CAP = None
OUTPUT_TRANSFORM = None
# Configure paths or environment variables
MODEL_DIR = "./models/"
DATASET_DIR = "./face_recognition/python/datasets/"


def build_argparser():
    parser = ArgumentParser()

    general = parser.add_argument_group("General")
    general.add_argument(
        "-i",
        "--input",
        required=False,
        default=0,
        help="Required. An input to process. The input must be a single image, "
        "a folder of images, video file or camera id.",
    )
    general.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Optional. Enable reading the input in a loop.",
    )
    general.add_argument(
        "-o",
        "--output",
        help="Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086",
    )
    general.add_argument(
        "-limit",
        "--output_limit",
        default=1000,
        type=int,
        help="Optional. Number of frames to store in output. "
        "If 0 is set, all frames are stored.",
    )
    general.add_argument(
        "--output_resolution",
        default=None,
        type=resolution,
        help="Optional. Specify the maximum output window resolution "
        "in (width x height) format. Example: 1280x720. "
        "Input frame size used by default.",
    )
    general.add_argument(
        "--no_show", action="store_true", help="Optional. Don't show output."
    )
    general.add_argument(
        "--crop_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Crop the input stream to this resolution.",
    )
    general.add_argument(
        "--match_algo",
        default="HUNGARIAN",
        choices=("HUNGARIAN", "MIN_DIST"),
        help="Optional. Algorithm for face matching. Default: HUNGARIAN.",
    )
    general.add_argument(
        "-u",
        "--utilization_monitors",
        default="",
        type=str,
        help="Optional. List of monitors to show initially.",
    )

    gallery = parser.add_argument_group("Faces database")
    gallery.add_argument(
        "-fg", default=DATASET_DIR, help="Optional. Path to the face images directory."
    )
    gallery.add_argument(
        "--run_detector",
        action="store_true",
        help="Optional. Use Face Detection model to find faces "
        "on the face images, otherwise use full images.",
    )
    gallery.add_argument(
        "--allow_grow",
        action="store_true",
        help="Optional. Allow to grow faces gallery and to dump on disk. "
        "Available only if --no_show option is off.",
    )

    models = parser.add_argument_group("Models")
    models.add_argument(
        "-m_FA", type=Path, required=False, default=MODEL_DIR + "anti_spoofing_480"
    )
    models.add_argument(
        "-m_fd",
        type=Path,
        required=False,
        default=MODEL_DIR + "face-detection-retail-0005.xml",
        help="Required. Path to an .xml file with Face Detection model.",
    )
    models.add_argument(
        "-m_lm",
        type=Path,
        required=False,
        default=MODEL_DIR + "landmarks-regression-retail-0009.xml",
        help="Required. Path to an .xml file with Facial Landmarks Detection model.",
    )
    models.add_argument(
        "-m_reid",
        type=Path,
        required=False,
        default=MODEL_DIR + "face-reidentification-retail-0095.xml",
        help="Required. Path to an .xml file with Face Reidentification model.",
    )
    models.add_argument(
        "--fd_input_size",
        default=(0, 0),
        type=int,
        nargs=2,
        help="Optional. Specify the input size of detection model for "
        "reshaping. Example: 500 700.",
    )

    infer = parser.add_argument_group("Inference options")
    infer.add_argument(
        "-d_fd",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Detection model. "
        "Default value is CPU.",
    )
    infer.add_argument(
        "-d_lm",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Facial Landmarks Detection "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-d_reid",
        default="CPU",
        choices=DEVICE_KINDS,
        help="Optional. Target device for Face Reidentification "
        "model. Default value is CPU.",
    )
    infer.add_argument(
        "-v", "--verbose", action="store_true", help="Optional. Be more verbose."
    )
    infer.add_argument(
        "-t_fd",
        metavar="[0..1]",
        type=float,
        default=0.6,
        help="Optional. Probability threshold for face detections.",
    )
    infer.add_argument(
        "-t_id",
        metavar="[0..1]",
        type=float,
        default=0.3,
        help="Optional. Cosine distance threshold between two vectors "
        "for face identification.",
    )
    infer.add_argument(
        "-exp_r_fd",
        metavar="NUMBER",
        type=float,
        default=1.15,
        help="Optional. Scaling ratio for bboxes passed to face recognition.",
    )
    return parser


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args):
        self.allow_grow = args.allow_grow and not args.no_show
        core = Core()

        self.face_detector = FaceDetector(
            core,
            args.m_fd,
            args.fd_input_size,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )

        self.face_anti_spoof = FaceAntiSpoof(core, args.m_FA)

        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(
            core, args.m_reid, match_threshold=args.t_id, match_algo=args.match_algo
        )

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug("Building faces database using images from {}".format(args.fg))
        self.faces_database = FacesDatabase(
            args.fg,
            self.face_identifier,
            self.landmarks_detector,
            self.face_detector if args.run_detector else None,
            args.no_show,
        )
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is built, registered {} identities".format(
                len(self.faces_database)
            )
        )
    
    def process(self, frame):
        orig_image = frame.copy()
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            rois = rois[: self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        return [rois, landmarks, face_identities]


def process_spoofing_result(result):
    # This function should process the spoofing result and convert it to a meaningful text
    # Example: if result is a probability or a binary classification
    if result > 6:
        return "Real"
    else:
        return "Spoof"
def draw_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    text = None
    for roi, landmarks, identity in zip(*detections):
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        face_img = frame[ymin:ymax, xmin:xmax]

        # Perform anti-spoofing inference on the face region
        spoofing_result = frame_processor.face_anti_spoof.infer(face_img)
        text2 = process_spoofing_result(spoofing_result)

        if text2 == "Real":
            text = frame_processor.face_identifier.get_identity_label(identity.id)
            confidence = 100.0 * (1 - identity.distance)
            if identity.id != FaceIdentifier.UNKNOWN_ID and confidence > 81.9:
                text += " %.2f%%" % confidence
            else:
                text = "Unknown"
        else:
            
            text = "Spoof"

        print(text2)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)

        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(
            frame,
            (xmin, ymin),
            (xmin + textsize[0], ymin - textsize[1]),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(
            frame, f"{text}_{text2}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
        )

    return frame, text


@app.route("/")
def index():
    return render_template("index.html")

def face_cropped(frame_processor, frame):
    detections = frame_processor.process(frame)
    if not detections or len(detections[0]) == 0:
        return None
    roi = detections[0][0]
    xmin = max(int(roi.position[0]), 0)
    ymin = max(int(roi.position[1]), 0)
    xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
    ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])
    cropped_face = frame[ymin:ymax, xmin:xmax]
    return cropped_face

def generate_dataset(frame_processor, frame, output_folder, count):
    cropped_face = face_cropped(frame_processor, frame)
    if cropped_face is not None:
        face = cv2.resize(cropped_face, (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name_path = os.path.join(output_folder, f"user_{count}.jpg")
        cv2.imwrite(file_name_path, face)
        return True
    return False

@app.route("/create-dataset", methods=["POST"])
def create_dataset():
    try:
        folder_name = request.args.get("folder_name")
        num_samples = int(request.args.get("num_samples", 100))
        output_folder = os.path.join(DATASET_DIR, folder_name)
        cap = cv2.VideoCapture(0)
        count = 0
        listData = []

        args = build_argparser().parse_args()
        frame_processor = FrameProcessor(args)

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if generate_dataset(frame_processor, frame, output_folder, count + 1):
                count += 1
                listData.append(f"user_{count}.jpg")

        cap.release()
        return jsonify({"status": 200, "message": "Create dataset successfully", "list": listData}), 200
    except Exception as e:
        return jsonify({"status": 400, "message": f"Error creating dataset {e}"}), 400
@app.route("/async-dataset", methods=["POST"])
def upload():
    global FRAME_PROCESSOR
    try:
        args = build_argparser().parse_args()
        FRAME_PROCESSOR = FrameProcessor(args)
        return jsonify({"success": "SYNC MODEL SUCCESSFULLY"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("connect")
def connect():
    global CAP, FRAME_PROCESSOR, OUTPUT_TRANSFORM
    CAP = cv2.VideoCapture(0)
    args = build_argparser().parse_args()
    FRAME_PROCESSOR = FrameProcessor(args)
    print(FRAME_PROCESSOR.face_anti_spoof)
    ret, frame = CAP.read()
    if ret:
        OUTPUT_TRANSFORM = OutputTransform(frame.shape[:2], None)

    emit("response", {"data": "Connected"})


@socketio.on("disconnect")
def disconnect():
    global CAP
    if CAP:
        CAP.release()
    emit("response", {"data": "Disconnected"})


last_emit_time = 0


@socketio.on("request_frame")
def handle_message():
    global CAP, FRAME_PROCESSOR, OUTPUT_TRANSFORM, last_emit_time

    ret, frame = CAP.read()
    if ret:
        detections = FRAME_PROCESSOR.process(frame)
        if len(detections) > 0:  # Only emit if there are detections

            frame, text = draw_detections(
                frame, FRAME_PROCESSOR, detections, OUTPUT_TRANSFORM
            )
            # print(text)
       
            _, buffer = cv2.imencode(".jpg", frame)
            frame_encoded = buffer.tobytes()
            emit("frame", {"frame_encoded": frame_encoded, "name": text})

        else:
            print("No detections, not emitting frame.")


if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
