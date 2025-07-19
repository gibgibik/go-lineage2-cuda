import cv2
import numpy as np
from flask import Flask, request, jsonify
import math
from imutils.object_detection import non_max_suppression
app = Flask(__name__)

EXCLUDE_BOUNDS = [
    (0, 0, 247, 104), # ex player stat
    (0, 590, 370, 1074), #  chat
    (697, 915, 1273, 1074), #panel with skills
    (1710, 0, 1920, 233), # map
    (1644, 0, 1748, 35), #money
    (902, 421, 1109, 665),#me
]

THRESHOLD = 0.9995
NMS_THRESHOLD = 0.4
RESIZE_W, RESIZE_H = 1920, 1088
rW = 1920 / RESIZE_W
rH = 1080 / RESIZE_H


net = None

@app.route("/findBounds", methods=["POST"])
def find_bounds():
   global net
   img_bytes = request.data
   npimg = np.frombuffer(img_bytes, np.uint8)
   image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
   orig = image.copy()
   (origH, origW) = image.shape[:2]
   newW = 1920
   newH = 1088
   image = cv2.resize(image, (newW, newH))
   (H, W) = image.shape[:2]
   blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
   	(123.68, 116.78, 103.94), swapRB=True, crop=False)
   net.setInput(blob)
   layerNames = [
   	"feature_fusion/Conv_7/Sigmoid",
   	"feature_fusion/concat_3"]

   (scores, geometry) = net.forward(layerNames)
   (rects, confidences) = decode_predictions(scores, geometry)
   boxes = non_max_suppression(np.array(rects), probs=confidences)
   if not rects:
       return jsonify({"boxes": []})

   # initialize the list of results
   results = []
   for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
#         dX = int((endX - startX) * 0.01)
#         dY = int((endY - startY) * 0.01)
#
#         # apply padding to each side of the bounding box, respectively
#         startX = max(0, startX - dX)
#         startY = max(0, startY - dY)
#         endX = min(origW, endX + (dX * 2))
#         endY = min(origH, endY + (dY * 2))
#
#         startX = startX - 10
#         startY = startY - 5
#         endX = endX + 10
#         endY = endY + 10

        if not check_excluded((startX, startY, endX, endY)):
            continue

        # extract the actual padded ROI
        if white_pixel_percent(orig, (startX, startY, endX, endY)) < 10:
            continue
        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
#         config = ("-l eng --oem 1 --psm 7")
#         text = pytesseract.image_to_string(roi, config=config)

        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY)))
   grouped = group_and_sort_rects(results)
   merged = []
   for g in grouped:
       merged.extend(merge_close_rects(g))

   return jsonify({"boxes": merged})
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.9995:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def check_excluded(rect):
    x1, y1, x2, y2 = rect
    for ex in EXCLUDE_BOUNDS:
        ex_x1, ex_y1, ex_x2, ex_y2 = ex

        # перевірка перетину двох прямокутників
        inter_x1 = max(x1, ex_x1)
        inter_y1 = max(y1, ex_y1)
        inter_x2 = min(x2, ex_x2)
        inter_y2 = min(y2, ex_y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            return False  # є перетин — виключити
    return True  # не перетинається з жодним — залишити


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

def group_and_sort_rects(rects, y_tol=3):
    """Групує прямокутники в рядки по Y."""
    groups = []
    for rect in rects:
        y = rect[1]
        added = False
        for group in groups:
            if abs(group[0][1] - y) <= y_tol:
                group.append(rect)
                added = True
                break
        if not added:
            groups.append([rect])
    # Сортування в групах за X
    for group in groups:
        group.sort(key=lambda r: r[0])
    # Сортування груп за Y
    groups.sort(key=lambda g: g[0][1])
    return groups

def merge_close_rects(line, x_tol=10):
    """Об'єднує сусідні прямокутники в лінії, якщо близько по X."""
    if not line:
        return []
    merged = [line[0]]
    for rect in line[1:]:
        last = merged[-1]
        if rect[0] - last[2] <= x_tol:
            merged[-1] = [
                min(last[0], rect[0]),
                min(last[1], rect[1]),
                max(last[2], rect[2]),
                max(last[3], rect[3]),
            ]
        else:
            merged.append(rect)
    return merged
if __name__ == "__main__":
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
#         net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#         net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    app.run(host="0.0.0.0", port=2224)
