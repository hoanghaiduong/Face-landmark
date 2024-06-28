"""
 Copyright (c) 2018-2024 Intel Corporation

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

import logging as log
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine

from face_detector import FaceDetector


class FacesDatabase:
    IMAGE_EXTENSIONS = ['jpg', 'png']

    class Identity:
        def __init__(self, label, descriptors):
            self.label = label
            self.descriptors = descriptors

        @staticmethod
        def cosine_dist(x, y):
            # cosine() returns 1 - cosine_similarity. cosine_similarity
            # belongs to the interval
            # [-1, 1] (https://en.wikipedia.org/wiki/Cosine_similarity).
            # (1 - cosine_similarity) belongs to the interval [0, 2].
            # To provide a probability like interpretation of the
            # similarity measure, the interval is scaled down by the
            # factor of two.
            return cosine(x, y) * 0.5

    # def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False):
        
    #     path = osp.abspath(path)
    #     self.fg_path = path
    #     self.no_show = no_show
    #     paths = []
    #     if osp.isdir(path):
    #         paths = [osp.join(path, f) for f in os.listdir(path)
    #                   if f.split('.')[-1] in self.IMAGE_EXTENSIONS]
    #     else:
    #         log.error("Wrong face images database path. Expected a "
    #                   "path to the directory containing %s files, "
    #                   "but got '%s'" %
    #                   (" or ".join(self.IMAGE_EXTENSIONS), path))

    #     if len(paths) == 0:
    #         log.error("The images database folder has no images.")

    #     self.database = []
    #     for path in paths:
    #         label = osp.splitext(osp.basename(path))[0]
    #         image = cv2.imread(path, flags=cv2.IMREAD_COLOR)

    #         orig_image = image.copy()

    #         if face_detector:
    #             rois = face_detector.infer((image,))
    #             if len(rois) < 1:
    #                 log.warning("Not found faces on the image '{}'".format(path))
    #         else:
    #             w, h = image.shape[1], image.shape[0]
    #             rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

    #         for roi in rois:
    #             r = [roi]
    #             landmarks = landmarks_detector.infer((image, r))

    #             face_identifier.start_async(image, r, landmarks)
    #             descriptor = face_identifier.get_descriptors()[0]

    #             if face_detector:
    #                 mm = self.check_if_face_exist(descriptor, face_identifier.get_threshold())
    #                 if mm < 0:
    #                     crop = orig_image[int(roi.position[1]):int(roi.position[1]+roi.size[1]),
    #                            int(roi.position[0]):int(roi.position[0]+roi.size[0])]
    #                     name = self.ask_to_save(crop)
    #                     self.dump_faces(crop, descriptor, name)
    #             else:
    #                 log.debug("Adding label {} to the gallery".format(label))
    #                 self.add_item(descriptor, label)
    def __init__(self, path, face_identifier, landmarks_detector, face_detector=None, no_show=False, load_from_pkl=False):
            self.database = []
            self.load_from_pkl = load_from_pkl
            self.fg_path = path
            self.face_identifier = face_identifier
            self.landmarks_detector = landmarks_detector
            self.face_detector = face_detector
            self.no_show = no_show

            path = Path(path)
            if not path.exists():
                raise ValueError("The images database folder does not exist")

            # Chỉ đọc ảnh nếu không load từ file .pkl
            if not self.load_from_pkl: 
                for person_dir in path.iterdir():
                    if person_dir.is_dir():
                        person_name = person_dir.name.split("_")[0]
                        for image_path in person_dir.iterdir():
                            if image_path.suffix.lower()[1:] in self.IMAGE_EXTENSIONS:
                                print(image_path)
                                self.add_image(person_name, str(image_path))

                if not self.database:
                    raise ValueError("The images database folder has no images")
    def add_new_face(self, folder_name):
        folder_path = os.path.join(self.fg_path, folder_name)
        if not os.path.isdir(folder_path):
            log.error(f"Folder '{folder_name}' does not exist in '{self.fg_path}'")
            return
        #image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        # for image_file in image_files:
        #     image_path = os.path.join(folder_path, image_file)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                log.warning(f"Cannot read image {image_path}")
                continue

            rois = self.face_detector.infer((image,))
            for roi in rois:
                r = [roi]
                landmarks = self.landmarks_detector.infer((image, r))

                self.face_identifier.start_async(image, r, landmarks)
                descriptor = self.face_identifier.get_descriptors()[0]

                if self.face_detector:
                    match = self.check_if_face_exist(descriptor, self.face_identifier.get_threshold())
                    if match < 0:
                        crop = image[int(roi.position[1]):int(roi.position[1]+roi.size[1]),
                                     int(roi.position[0]):int(roi.position[0]+roi.size[0])]
                        name = self.ask_to_save(crop)
                        self.dump_faces(crop, descriptor, name, folder_name)
                else:
                    log.debug("Adding new face to the database")
                    label = os.path.splitext(filename)[0]
                    self.add_item(descriptor, label)

    def add_image(self, label, image_path):
        image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        if image is None:
            log.error(f"Cannot read image {image_path}")
            return

        orig_image = image.copy()

        if self.face_detector:
            rois = self.face_detector.infer((image,))
            if len(rois) < 1:
                log.warning(f"Not found faces on the image '{image_path}'")
                return
        else:
            w, h = image.shape[1], image.shape[0]
            rois = [FaceDetector.Result([0, 0, 0, 0, 0, w, h])]

        for roi in rois:
            r = [roi]
            landmarks = self.landmarks_detector.infer((image, r))
            self.face_identifier.start_async(image, r, landmarks)
            descriptor = self.face_identifier.get_descriptors()[0]
            self.add_item(descriptor, label)

    def ask_to_save(self, image):
        if self.no_show:
            return None
        save = False
        winname = "Unknown face"
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 0, 0)
        w = int(400 * image.shape[0] / image.shape[1])
        sz = (400, w)
        resized = cv2.resize(image, sz, interpolation=cv2.INTER_AREA)
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 1
        img = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(img, 'This is an unrecognized image.', (30, 50), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'If you want to store it to the gallery,', (30, 80), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'please, put the name and press "Enter".', (30, 110), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'Otherwise, press "Escape".', (30, 140), font, fontScale, fontColor, lineType)
        cv2.putText(img, 'You can see the name here:', (30, 170), font, fontScale, fontColor, lineType)
        name = ''
        while 1:
            cc = img.copy()
            cv2.putText(cc, name, (30, 200), font, fontScale, fontColor, lineType)
            cv2.imshow(winname, cc)

            k = cv2.waitKey(0)
            if k == 27: #Esc
                break
            if k == 13: #Enter
                if len(name) > 0:
                    save = True
                    break
                else:
                    cv2.putText(cc, "Name was not inserted. Please try again.", (30, 200), font, fontScale, fontColor, lineType)
                    cv2.imshow(winname, cc)
                    k = cv2.waitKey(0)
                    if k == 27:
                        break
                    continue
            if k == 225: #Shift
                continue
            if k == 8: #backspace
                name = name[:-1]
                continue
            else:
                name += chr(k)
                continue

        cv2.destroyWindow(winname)
        return name if save else None

    def match_faces(self, descriptors, match_algo='HUNGARIAN'):
        distances = np.empty((len(descriptors), len(self.database)))
        for i, desc in enumerate(descriptors):
            for j, identity in enumerate(self.database):
                dist = [FacesDatabase.Identity.cosine_dist(desc, id_desc) for id_desc in identity.descriptors]
                distances[i][j] = min(dist)

        matches = []
        
        if match_algo == 'MIN_DIST':
            for i in range(len(descriptors)):
                id = np.argmin(distances[i])
                min_dist = distances[i][id]
                matches.append((id, min_dist))
        else:
            row_ind, col_ind = linear_sum_assignment(distances)
            for i in range(len(descriptors)):
                if i < len(row_ind):
                    id = col_ind[i]
                    distance = distances[row_ind[i], id]
                    matches.append((id, distance))
                else:
                    matches.append((-1, float('inf')))  # No match found

        return matches


    def create_new_label(self, path, id):
        while osp.exists(osp.join(path, "face{}.jpg".format(id))):
            id += 1
        return "face{}".format(id)

    def check_if_face_exist(self, desc, threshold):
        match = -1
        for j, identity in enumerate(self.database):
            dist = []
            for id_desc in identity.descriptors:
                dist.append(FacesDatabase.Identity.cosine_dist(desc, id_desc))
            if dist[np.argmin(dist)] < threshold:
                match = j
                break
        return match

    def check_if_label_exists(self, label):
        match = -1
        import re
        name = re.split(r'-\d+$', label)
        if not len(name):
            return -1, label
        label = name[0].lower()

        for j, identity in enumerate(self.database):
            if identity.label == label:
                match = j
                break
        return match, label

    def dump_faces(self, image, desc, name, folder_name):
        folder_path = os.path.join(self.fg_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Generate unique filename based on existing images in the folder
        existing_files = os.listdir(folder_path)
        next_id = len(existing_files) + 1
        filename = f"{name}_{next_id}.jpg"
        
        file_path = os.path.join(folder_path, filename)
        log.debug(f"Dumping image with label {name} and path {file_path} on disk.")
        cv2.imwrite(file_path, image)

        # Add the descriptor to the database
        self.add_item(desc, name)

    def add_item(self, desc, label):
        match = -1
        if not label:
            label = self.create_new_label(self.fg_path, len(self.database))
            log.warning("Trying to store an item without a label. Assigned label {}.".format(label))
        else:
            match, label = self.check_if_label_exists(label)

        if match < 0:
            self.database.append(self.Identity(label, [desc]))
        else:
            self.database[match].descriptors.append(desc)
            log.debug("Appending new descriptor for label {}.".format(label))

        return match, label


    def __getitem__(self, idx):
        return self.database[idx]

    def __len__(self):
        return len(self.database)
