from toolkit.motion_detection import SingleMotionDetector
from toolkit.parseyolooutput import ParseYOLOOutput
from toolkit.keyclipwriter import KeyClipWriter
from toolkit.utils.conf import Conf
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, 
	help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the ImageHub object
conf = Conf(args["config"])
imageHub = imagezmq.ImageHub()

# initialize the motion detector 
# total number of frames read thus far
# spatial dimensions of the frame
md = SingleMotionDetector(accumWeight=0.1)
total = 0
(W, H) = (None, None)

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([conf["yolo_path"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([conf["yolo_path"], "yolov3.weights"])
configPath = os.path.sep.join([conf["yolo_path"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# determine only the output layer names that we need from YOLO
print("Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the YOLO output parsing object
pyo = ParseYOLOOutput(conf)

# initialize key clip writer
# consecutive number of frames that have not contained any action
kcw = KeyClipWriter(bufSize=conf["buffer_size"])
consecFrames = 0
print("Starting surveillance...")

# start looping over all the frames
while True:
	print('enter')
	# receive client name and frame
	# acknowledge the receipt
	(clientName, frame) = imageHub.recv_image()
	print('got {}'.format(clientName))
	imageHub.send_reply(b'Status: 200')
	print('sent reply')

	# resize the frame
	# convert it to grayscale
	# blur it to reduce noise
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	# grab the current timestamp and draw it on the frame
	timestamp = datetime.now()
	cv2.putText(frame, timestamp.strftime(
		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	# if we do not already have the dimensions of the frame,
	# initialize it
	if H is None and W is None:
		(H, W) = frame.shape[:2]

	# if the total number of frames has reached a sufficient
	# number to construct a background model
	# process the frame
	if total > conf["frame_count"]:
		# detect motion in the frame and set the update consecutive
		# frames flag as True
		motion = md.detect(gray)
		updateConsecFrames = True

		# if the motion object is not None, then motion has
		# occurred in the image
		if motion is not None:
			# set the update consecutive frame flag as false and
			# reset the number of consecutive frames with no action
			# to zero
			updateConsecFrames = False
			consecFrames = 0

			# if we are not already recording, start recording
			if not kcw.recording:
				# store the day's date and check if output directory
				# exists, or create it
				date = timestamp.strftime("%Y-%m-%d")
				os.makedirs(os.path.join(conf["output_path"], date),
					exist_ok=True)

				# build the output video path and start recording
				p = "{}/{}/{}.avi".format(conf["output_path"], date,
					timestamp.strftime("%H%M%S"))
				kcw.start(p, cv2.VideoWriter_fourcc(*conf["codec"]),
					conf["fps"])

			# construct a blob from the input frame and then perform
			# a forward pass of the YOLO object detector, giving us
			# our bounding boxes and associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0,
				(416, 416), swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# parse YOLOv3 output
			(boxes, confidences, classIDs) = pyo.parse(layerOutputs,
				LABELS, H, W)

			# apply non-maxima suppression to suppress weak,
			# overlapping bounding boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences,
				conf["confidence"], conf["threshold"])

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					# draw a bounding box rectangle and label on the
					# frame
					color = [int(c) for c in COLORS[classIDs[i]]]
					cv2.rectangle(frame, (x, y), (x + w, y + h),
						color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]],
						confidences[i])
					y = (y - 15) if (y - 15) > 0 else h - 15
					cv2.putText(frame, text, (x, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		# otherwise, no action has taken place in this frame, so
		# increment the number of consecutive frames that contain
		# no action
		if updateConsecFrames:
			consecFrames += 1

		# update the key frame clip buffer
		kcw.update(frame)

		# if we are recording and reached a threshold on consecutive
		# number of frames with no action, stop recording the clip
		if kcw.recording and consecFrames == conf["buffer_size"]:
			kcw.finish()

	# update the background model and increment the total number
	# of frames read thus far
	md.update(gray)
	total += 1

	# show the frame
	cv2.imshow("{}".format(clientName), frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording:
	kcw.finish()

# do a bit of cleanup
cv2.destroyAllWindows()