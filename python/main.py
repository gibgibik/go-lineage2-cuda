import cv2
import numpy as np
from flask import Flask, request, jsonify
import math

app = Flask(__name__)

EXCLUDE_BOUNDS = [
    (0, 0, 247, 104),
    (0, 590, 370, 1074),
    (697, 915, 1273, 1074),
    (1710, 0, 1920, 350),
    (1644, 0, 1748, 35),
    (902, 478, 1040, 649),
]

THRESHOLD = 0.9995
NMS_THRESHOLD = 0.4
RESIZE_W, RESIZE_H = 1920, 1088
rW = 1920 / RESIZE_W
rH = 1080 / RESIZE_H


net = None

@app.route("/findBounds", methods=["POST"])
def find_bounds():
    global net  # доступ до глобальної змінної

    img_bytes = request.data
    npimg = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#     net = cv2.dnn.readNet("frozen_east_text_detection.pb")
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(image, 1.0, (RESIZE_W, RESIZE_H), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    boxes, confidences = decode(scores, geometry)
    rotated_boxes = [((b[0], b[1]), (b[2], b[3]), b[4]) for b in boxes]
    indices = cv2.dnn.NMSBoxesRotated(rotated_boxes, confidences, THRESHOLD, NMS_THRESHOLD)

    final = []
    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x, y, w, h, angle = boxes[i]
        rect = cv2.boxPoints(((x, y), (w, h), angle))
        rect = np.intp(rect * np.array([rW, rH]))
        x1, y1 = np.min(rect, axis=0)
        x2, y2 = np.max(rect, axis=0)
        if not check_excluded((x1, y1, x2, y2)):
            continue
        if white_pixel_percent(image, (x1, y1, x2, y2)) < 10:
            continue
        final.append([int(x1), int(y1), int(x2), int(y2)])

    grouped = group_and_sort(final)
    merged = []
    for g in grouped:
        merged.extend(merge_line(g))

    return jsonify({"boxes": merged})


def decode(scores, geometry):
    boxes = []
    confidences = []
    height, width = scores.shape[2:4]
    for y in range(height):
        for x in range(width):
            score = scores[0, 0, y, x]
            if score < THRESHOLD:
                continue
            angle = geometry[0, 4, y, x]
            cos = math.cos(angle)
            sin = math.sin(angle)
            x0 = geometry[0, 0, y, x]
            x1 = geometry[0, 1, y, x]
            x2 = geometry[0, 2, y, x]
            x3 = geometry[0, 3, y, x]

            offsetX, offsetY = x * 4.0, y * 4.0
            h = x0 + x2
            w = x1 + x3
            endX = int(offsetX + cos * x1 + sin * x2)
            endY = int(offsetY - sin * x1 + cos * x2)
            boxes.append([endX, endY, int(w), int(h), -angle * 180 / math.pi])
            confidences.append(float(score))
    return boxes, confidences


def check_excluded(rect):
    x1, y1, x2, y2 = rect
    for ex in EXCLUDE_BOUNDS:
        if ex[0] <= x1 and ex[1] <= y1 and ex[2] >= x2 and ex[3] >= y2:
            return False
    return True


def white_pixel_percent(img, rect):
    x1, y1, x2, y2 = rect
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    return 100.0 * cv2.countNonZero(mask) / (gray.shape[0] * gray.shape[1])


def group_and_sort(rects):
    y_tol = 3
    groups = []
    for r in rects:
        y = r[1]
        found = False
        for g in groups:
            if abs(g[0][1] - y) <= y_tol:
                g.append(r)
                found = True
                break
        if not found:
            groups.append([r])
    for g in groups:
        g.sort(key=lambda r: r[0])  # X1
    groups.sort(key=lambda g: g[0][1])  # Y1
    return groups


def merge_line(line):
    x_tol = 10
    if not line:
        return []
    merged = [line[0]]
    for r in line[1:]:
        last = merged[-1]
        if r[0] - last[2] <= x_tol:
            merged[-1] = [
                min(last[0], r[0]),
                min(last[1], r[1]),
                max(last[2], r[2]),
                max(last[3], r[3]),
            ]
        else:
            merged.append(r)
    return merged


if __name__ == "__main__":
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except Exception as e:
        print("⚠️ CUDA не доступна, перемикаюсь на CPU:", e)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    app.run(host="0.0.0.0", port=2224)
