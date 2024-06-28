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
import pickle
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
#DATASET_DIR = "./datasets/"
dataset_ready = False
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
        self.embeddings = []
        self.names = []
        
        self.allow_grow = args.allow_grow and not args.no_show
        core = Core()

        self.face_detector = FaceDetector(
            core,
            args.m_fd,
            args.fd_input_size,
            confidence_threshold=args.t_fd,
            roi_scale_factor=args.exp_r_fd,
        )


        self.landmarks_detector = LandmarksDetector(core, args.m_lm)
        self.face_identifier = FaceIdentifier(
            core, args.m_reid, match_threshold=args.t_id, match_algo=args.match_algo
        )

        self.face_detector.deploy(args.d_fd)
        self.landmarks_detector.deploy(args.d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(args.d_reid, self.QUEUE_SIZE)

        log.debug("Building faces database using images from {}".format(args.fg))
        if args.load_from_pkl:
            self.faces_database = FacesDatabase(
                args.fg,
                self.face_identifier,
                self.landmarks_detector,
                self.face_detector if args.run_detector else None,
                args.no_show,
                args.load_from_pkl,
            )
        else:
            self.faces_database = FacesDatabase(
                args.fg,
                self.face_identifier,
                self.landmarks_detector,
                self.face_detector if args.run_detector else None,
                args.no_show,
            )
        pkl_path = "face_data.pkl" # Đường dẫn đến file .pkl

        if args.load_from_pkl:
            if not self._load_from_pkl(pkl_path):
                # Xử lý trường hợp load từ file thất bại
                # Ví dụ: 
                # print("Failed to load")
                log.warning("Loading from .pkl failed, building database from images...")
                self._build_database_from_images(args) 
           
        else:
            self._build_database_from_images(args)
            self._save_to_pkl(pkl_path)
    
    def _load_from_pkl(self, pkl_path):
        """Nạp dữ liệu từ file .pkl."""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            self.faces_database.database = [
                FacesDatabase.Identity(label, descriptors)
                for label, descriptors in zip(data['names'], data['embeddings'])
            ]
            self.face_identifier.set_faces_database(self.faces_database)
            log.info(f"Data loaded successfully from {pkl_path}")
            return True
        except Exception as e:
            log.warning(f"Error loading data from {pkl_path}: {e}")
            return False
    def _build_database_from_images(self, args):
        self.face_identifier.set_faces_database(self.faces_database)
        log.info(
            "Database is built, registered {} identities".format(
                len(self.faces_database)
            )
        )
    def _save_to_pkl(self, pkl_path):
        """Lưu dữ liệu vào file .pkl."""
        try:
            data = {
                'names': [identity.label for identity in self.faces_database.database],
                'embeddings': [identity.descriptors for identity in self.faces_database.database]
            }
            with open(pkl_path, 'wb') as f:
                pickle.dump(data, f)
            log.info(f"Data saved successfully to {pkl_path}")
        except Exception as e:
            log.warning(f"Error saving data to {pkl_path}: {e}")
    # def save_embeddings(self, file_path="face_data.pkl"):
    #     """Lưu embeddings vào file pickle."""
    #     data = {
    #         "embeddings": self.embeddings,
    #         "names": self.names,
    #     }
    #     with open(file_path, "wb") as f:
    #         pickle.dump(data, f)
    #     log.info(f"Embeddings saved successfully to {file_path}")
        
    # def _extract_dataset_embeddings(self):
    #     """Trích xuất embeddings và names cho toàn bộ dataset."""
    #     self.embeddings = []
    #     self.names = []

    #     for identity in self.faces_database.database:
    #         person_name = identity.label
            
    #         # Lấy descriptors từ identity
    #         for descriptor in identity.descriptors:
    #             self.embeddings.append(descriptor)
    #             self.names.append(person_name)
    def process(self, frame):
        orig_image = frame.copy()
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            rois = rois[: self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))
        # self.embeddings = []  # Khởi tạo lại list embeddings
        # self.names = []
        # print(face_identities[0])  # In ra đối tượng Result đầu tiên
        # print(dir(face_identities[0]))  # Liệt kê các thuộc tính và phương thức của Result
        # for identity in face_identities:  # Duyệt qua từng identity
        #     self.embeddings.append(identity.descriptor)

        #     name = self.face_identifier.get_identity_label(identity.id)
        #     confidence = 100.0 * (1 - identity.distance)
        #     if identity.id != FaceIdentifier.UNKNOWN_ID and confidence > 81.9:
        #         self.names.append(name)
        #     else:
        #         self.names.append("Unknown")
        # print(unknowns)
        # self.names = 
        return [rois, landmarks, face_identities]
    
    def process_not_reidentity(self, frame):
        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            rois = rois[: self.QUEUE_SIZE]
        landmarks = self.landmarks_detector.infer((frame, rois))
        return [rois, landmarks]

# def face_detector(frame,frame_processor,output_transform):
#     size = frame.shape[:2]
#     frame = output_transform.resize(frame)
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
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        confidence = 100.0 * (1 - identity.distance)
        if identity.id != FaceIdentifier.UNKNOWN_ID and confidence > 81.9:
            text += " %.2f%%" % confidence
        else:
            text = "Unknown"
        # Perform anti-spoofing inference on the face region

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
            frame,
            f"{text}",
            (xmin, ymin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
        )

    return frame, text

def draw_face_detections(frame, frame_processor, detections, output_transform):
    size = frame.shape[:2]
    frame = output_transform.resize(frame)

    for roi, landmarks in zip(*detections):
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), size[1])
        ymax = min(int(roi.position[1] + roi.size[1]), size[0])
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + output_transform.scale(roi.size[0] * point[0])
            y = ymin + output_transform.scale(roi.size[1] * point[1])
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 255), 2)

    return frame
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/trainning")
def trainning():
    return render_template("trainning.html")
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
        args.load_from_pkl = True
        frame_processor = FrameProcessor(args)

        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if generate_dataset(frame_processor, frame, output_folder, count + 1):
                count += 1
                listData.append(f"user_{count}.jpg")
        args.load_from_pkl = False
        frame_processor = FrameProcessor(args)
        cap.release()
        # frame_processor.save_embeddings()
        return (
            jsonify(
                {
                    "status": 200,
                    "message": "Create dataset successfully",
                    "list": listData,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"status": 400, "message": f"Error creating dataset {e}"}), 400


@socketio.on("connect")
def handle_connect():
    emit("response", {"data": "Connected"})
            


@app.route("/async-dataset", methods=["POST"])
def upload():
    global FRAME_PROCESSOR, OUTPUT_TRANSFORM, dataset_ready
    try:
        args = build_argparser().parse_args()
        args.load_from_pkl = True
        FRAME_PROCESSOR = FrameProcessor(args)
        # FRAME_PROCESSOR.load_embeddings()
        OUTPUT_TRANSFORM = OutputTransform((640, 480), None)  # Replace (640, 480) with actual dimensions
        dataset_ready = True
        return jsonify({"success": "SYNC MODEL SUCCESSFULLY"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

last_emit_time = 0

@socketio.on("request_frame")
def handle_message():
    global CAP, FRAME_PROCESSOR, OUTPUT_TRANSFORM, dataset_ready

    if CAP is None or not CAP.isOpened():
        CAP = cv2.VideoCapture(0)

    if CAP.isOpened():
        ret, frame = CAP.read()
        if ret:
            if not dataset_ready:
                _, buffer = cv2.imencode(".jpg", frame)
                frame_encoded = buffer.tobytes()
                emit("frame", {"frame_encoded": frame_encoded, "message": "Not dataset ready please async data transfer"})
            else:
                detections = FRAME_PROCESSOR.process(frame)
                if len(detections) > 0:  # Chỉ emit nếu có detections
                    frame, text = draw_detections(frame, FRAME_PROCESSOR, detections, OUTPUT_TRANSFORM)
                    _, buffer = cv2.imencode(".jpg", frame)
                    frame_encoded = buffer.tobytes()
                    emit("frame", {"frame_encoded": frame_encoded, "name": text})
                else:
                    print("No detections, not emitting frame.")
    else:
        emit("frame", {"message": "Unable to open camera"})


@socketio.on("disconnect")
def disconnect():
    global CAP
    if CAP:
        CAP.release()
    emit("response", {"data": "Disconnected"})

@socketio.on("request_frame_create_dataset")
def handle_start_trainning_frame():
    global CAP, FRAME_PROCESSOR, OUTPUT_TRANSFORM

    if CAP is None or not CAP.isOpened():
        CAP = cv2.VideoCapture(0)

    while CAP.isOpened():
        ret, frame = CAP.read()
        if ret:
            OUTPUT_TRANSFORM = OutputTransform(frame.shape[:2], None)
            detections = FRAME_PROCESSOR.process_not_reidentity(frame)
            frame = draw_face_detections(frame, FRAME_PROCESSOR, detections, OUTPUT_TRANSFORM)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_encoded = buffer.tobytes()
            emit("frame_trainning", {"frame_encoded": frame_encoded}, broadcast=True)
        else:
            break

@socketio.on("stop_frame_create_dataset")
def handle_stop_trainning_frame():
    global CAP
    if CAP:
        CAP.release()
    emit("response", {"data": "Camera stopped."})

if __name__ == "__main__":

    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
