# coding:utf-8
import os
import cv2
import logging
import numpy as np
import numpy.random as npr
from prepare_data.utils import IoU

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

LOGGER = logging.getLogger("gen_12net_data")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(console_handler)

WIDER_DIR = "/opt/data/wider"
ANNOTATION_FILE = os.path.join(WIDER_DIR, "wider_face_split/wider_face_train_bbx_gt.txt")
IMAGE_DIR = os.path.join(WIDER_DIR, "WIDER_train/images")

SAVE_DIR = os.path.join(WIDER_DIR, "save")
pos_save_dir = os.path.join(SAVE_DIR, "positive")
part_save_dir = os.path.join(SAVE_DIR, "part")
neg_save_dir = os.path.join(SAVE_DIR, "negative")


def check_data_path():
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.exists(pos_save_dir):
        os.mkdir(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.mkdir(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.mkdir(neg_save_dir)


def read_annotation_file():
    with open(ANNOTATION_FILE, 'r') as f:
        return f.readlines()


class FaceImage:
    positive_index = 0
    negative_index = 0
    part_index = 0

    f1 = open(os.path.join(SAVE_DIR, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(SAVE_DIR, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(SAVE_DIR, 'part_12.txt'), 'w')

    def __init__(self, filename):
        self.filename = filename
        self.bbox_num = 0
        self.finish = False
        self.bbox_list = []
        self.boxes = None

    def parse(self, line):
        line = line.split(' ')
        if self.bbox_num == 0:
            if len(line) == 1:
                self.bbox_num = int(line[0])
            else:
                LOGGER.warn("parse image bbox size error")
                self.finish = True
        else:
            self.bbox_list.append(list(map(float, line[0:4])))
        if self.is_finish():
            return self.gen_data()

    def is_finish(self):
        if self.finish:
            return True
        if self.bbox_num == 0:
            return False
        elif self.bbox_num != len(self.bbox_list):
            return False
        else:
            return True

    def gen_data(self):
        LOGGER.debug("generate image = %s training data" % self.filename, )
        self.boxes = np.array(self.bbox_list, dtype=np.float32).reshape(-1, 4)
        img = cv2.imread(os.path.join(IMAGE_DIR, self.filename))
        height, width, channel = img.shape
        self.gen_negative50_data(img, width, height)
        self.gen_data_from_boxes(img, width, height)

    def gen_negative50_data(self, img, width, height):
        neg_num = 0
        while neg_num < 50:
            size = npr.randint(12, min(width, height) / 2)
            nx = npr.randint(0, width - size)
            ny = npr.randint(0, height - size)
            crop_box = np.array([nx, ny, nx + size, ny + size])
            iou = IoU(crop_box, self.boxes)
            cropped_im = img[ny: ny + size, nx: nx + size, :]
            resize_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            if np.max(iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % FaceImage.negative_index)
                FaceImage.f2.write(save_file.replace(SAVE_DIR, "") + "\n")
                cv2.imwrite(save_file, resize_im)
                FaceImage.negative_index += 1
                neg_num += 1

    def gen_data_from_boxes(self, img, width, height):
        for box in self.boxes:
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            if max(w, h) < 20 or x1 < 0 or y1 < 0:
                continue
            self.gen_negative5_data(img, width, height, x1, y1, w, h)
            self.gen_positive_part_20_data(img, box, width, height, x1, y1, x2, y2, w, h)

    def gen_negative5_data(self, img, width, height, x1, y1, w, h):
        for i in range(5):
            size = npr.randint(12, min(width, height) / 2)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            iou = IoU(crop_box, self.boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resize_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % FaceImage.negative_index)
                FaceImage.f2.write(save_file.replace(SAVE_DIR, "") + "\n")
                cv2.imwrite(save_file, resize_im)
                FaceImage.negative_index += 1

    @staticmethod
    def gen_positive_part_20_data(img, box, width, height, x1, y1, x2, y2, w, h):
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            if w < 5:
                print (w)
                continue
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            # show this way: nx1 = max(x1+w/2-size/2+delta_x)
            # x1+ w/2 is the central point, then add offset , then deduct size/2
            # deduct size/2 to make sure that the right bottom corner will be out of
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resize_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % FaceImage.positive_index)
                FaceImage.f1.write(save_file.replace(SAVE_DIR, "") + ' 1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resize_im)
                FaceImage.positive_index += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % FaceImage.part_index)
                FaceImage.f3.write(save_file.replace(SAVE_DIR, "") + ' -1 %.2f %.2f %.2f %.2f\n' % (
                    offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resize_im)
                FaceImage.part_index += 1


if __name__ == "__main__":
    check_data_path()
    annotations = read_annotation_file()
    num = len(annotations)
    LOGGER.debug("annotations file count = %d" % num)

    faceImage = None
    for annotation in annotations:
        annotation = annotation.strip()
        if annotation.endswith("jpg"):
            faceImage = FaceImage(annotation)
        elif faceImage is not None:
            faceImage.parse(annotation)
            if faceImage.is_finish():
                faceImage = None
        else:
            LOGGER.info("parse error, ignore line = %s" % annotation)

