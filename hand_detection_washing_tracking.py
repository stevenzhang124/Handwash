#import matplotlib
#matplotlib.use('Agg')
from flask import Flask, render_template, Response
import sys
import time
import logging
import subprocess
import cv2

from collections import deque
from tracker import Tracker
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

from pedestrian_detection_ssdlite import api
from reid import cam_reid
from matplotlib import pyplot as plt

#for hand detection
from utils import detector_utils_washhand as detector_utils
import tensorflow as tf
import datetime
import argparse
import sqlite3

#set args for hand detection
global im_width
global im_height
global detection_graph
global sess


#load the hand detection graph and set arg for hand detection
detection_graph, sess = detector_utils.load_inference_graph()
num_hands_detect = 6
score_thresh = 0.2
sink_loc= [(440,359),(504,452)]
patient_loc= [(126,358),(226,481)]
im_width = 640
im_height = 480
thre_hp_match = 0.001 # threshold for hand and person BBox matching

# global variables to be used in the code for tracker
max_age=10
min_hits=1

app = Flask(__name__)
'''
logging.basicConfig(
	stream=sys.stdout,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	datefmt=' %I:%M:%S ',
	level="INFO"
)
logger = logging.getLogger('detector')
'''
'''
def open_cam_onboard(width, height):
	# On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
	gst_str = ('nvcamerasrc ! '
			   'video/x-raw(memory:NVMM), '
			   'width=(int)2592, height=(int)1458, '
			   'format=(string)I420, framerate=(fraction)30/1 ! '
			   'nvvidconv ! '
			   'video/x-raw, width=(int){}, height=(int){}, '
			   'format=(string)BGRx ! '
			   'videoconvert ! appsink').format(width, height)
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
'''

reid_mode = cam_reid.reid_model()

# encode origin image
compare = cam_reid.Compare(model=reid_mode, origin_img="./image/origin")
origin_f, origin_name = compare.encode_origin_image()

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])


def open_cam_onboard(width, height):
	gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
	if 'nvcamerasrc' in gst_elements:
		# On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
		gst_str = ('nvcamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)2592, height=(int)1458, '
				   'format=(string)I420, framerate=(fraction)30/1 ! '
				   'nvvidconv ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	elif 'nvarguscamerasrc' in gst_elements:
		gst_str = ('nvarguscamerasrc ! '
				   'video/x-raw(memory:NVMM), '
				   'width=(int)1920, height=(int)1080, '
				   'format=(string)NV12, framerate=(fraction)30/1 ! '
				   'nvvidconv flip-method=0 ! '
				   'video/x-raw, width=(int){}, height=(int){}, '
				   'format=(string)BGRx ! '
				   'videoconvert ! appsink').format(width, height)
	else:
		raise RuntimeError('onboard camera source not found!')
	return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[1][0], b[1][0]) - np.maximum(a[0][0], b[0][0])))
    h_intsec = np.maximum (0, (np.minimum(a[1][1], b[1][1]) - np.maximum(a[0][1], b[0][1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[1][0] - a[0][0])*(a[1][1] - a[0][1])
    s_b = (b[1][0] - b[0][0])*(b[1][1] - b[0][1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx_tra, matched_idx_det = linear_assignment(-IOU_mat)        
    matched_idx = np.zeros((len(matched_idx_tra),2),dtype=np.int8)
    for i in range(len(matched_idx_tra)):
        matched_idx[i]=(matched_idx_tra[i],matched_idx_det[i])

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def wash_hand_detector(p_tracker, hand_in_sink):
	p_left, p_top, p_right, p_bottom = p_tracker.box[0], p_tracker.box[1], p_tracker.box[2], p_tracker.box[3]
	p_box = [(p_left, p_top),(p_right, p_bottom)]
	p_b_iou = box_iou2(p_box, hand_in_sink)
	person_hand_match = 0
	if p_b_iou > thre_hp_match:
		if (not p_tracker.hand_clean) :
			p_tracker.have_washed_hand = 1 #if the person washed hand, then his hand clean
			p_tracker.hand_clean = 1
			person_hand_match = 1

	return person_hand_match

def touch_patient_detector(p_tracker, hand_in_patient):
	p_left, p_top, p_right, p_bottom = p_tracker.box[0], p_tracker.box[1], p_tracker.box[2], p_tracker.box[3]
	#h_left, h_top, h_right, h_bottom = int(boxes[i][1] * im_width), int(boxes[i][0] * im_height), int(boxes[i][3] * im_width), int(boxes[i][2] * im_height)
	p_box = [(p_left, p_top),(p_right, p_bottom)]
	#h_box =  [(h_left, h_top),(h_right, h_bottom)]
	p_b_iou = box_iou2(p_box, hand_in_patient)
	person_hand_match = 0
	if p_b_iou > thre_hp_match:
		if p_tracker.have_washed_hand and p_tracker.hand_clean:
			p_tracker.have_touched_pat = 1 #if person has washed hand and his hand clean, the touch activate
			p_tracker.hand_clean = 0
			person_hand_match = 1
		if (not p_tracker.have_washed_hand) and (not p_tracker.have_touched_pat):
			p_tracker.have_touched_pat = 1 #if person hasn't washed hand, the touch activste, and he violate
			p_tracker.hand_clean = 0
			person_hand_match = 1
			p_tracker.violate_rule = 2


	return person_hand_match

def draw_box_label(img, bbox_cv2, box_color=(0, 0, 255), personReID_info={'personID':'Unknown'}, show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.6
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[0], bbox_cv2[1], bbox_cv2[2], bbox_cv2[3]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-30), (right+2, top), box_color, -1, 1)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        #text_x= 'x='+str((left+right)/2)
        text_x= ''
        cv2.putText(img,text_x,(left,top-20), font, font_size, font_color, 1, cv2.LINE_AA)  
        text_ID = personReID_info['personID']
        cv2.putText(img,text_ID,(left,top-10), font, font_size, font_color, 1, cv2.LINE_AA)            
        #text_y= 'y='+str((top+bottom)/2)
        text_y = ''
        cv2.putText(img,text_y,(left,top), font, font_size, font_color, 1, cv2.LINE_AA)
            
    return img    



def handle_frames(frame):

	global tracker_list
	global max_age
	global min_hits
	global track_id_list
	
	#connect to database
	conn = sqlite3.connect('handwash.db', isolation_level=None)
	#print("Opened database successfully")
	cur = conn.cursor()


	#detect hand
	try:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	except:
		print("Error converting to RGB")
	#print(type(frame))
	boxes, scores = detector_utils.detect_objects(frame,detection_graph,sess)

	# draw bounding boxes on frame
	hand_in_sink, hand_in_patient = detector_utils.draw_box_on_image_washhand( \
		num_hands_detect, score_thresh, scores, boxes, im_width, \
		im_height, frame, sink_loc, patient_loc)

	try:
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	except:
		print("Error converting to BGR")


	#detect person
	detection_results = api.get_person_bbox(frame, thr=0.50)
	x_box =[]
	if len(tracker_list) > 0:
		for trk in tracker_list:
			x_box.append([(trk.box[0],trk.box[1]),(trk.box[2],trk.box[3])]) #should be changed into the right format instead of the .box format

	matched, unmatched_dets, unmatched_trks = assign_detections_to_trackers(x_box, detection_results, iou_thrd = 0.2)  

	# Deal with matched detections     
	if matched.size >0:
		for trk_idx, det_idx in matched:
			z = detection_results[det_idx]
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk= tracker_list[trk_idx]
			tmp_trk.kalman_filter(z)
			xx = tmp_trk.x_state.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			x_box[trk_idx] = xx
			tmp_trk.box =xx
			tmp_trk.hits += 1
			tmp_trk.no_losses = 0

	# Deal with unmatched detections      
	if len(unmatched_dets)>0:
		for idx in unmatched_dets:
			z = detection_results[idx]
			x1 = int(z[0][0])
			y1 = int(z[0][1])
			x2 = int(z[1][0])
			y2 = int(z[1][1])
			person = frame[y1:y2, x1:x2, :]
			identify_name, score = compare.run(person, origin_f, origin_name)
			if(identify_name in [ "QY1", "QY2", "QY3", "QY4", "QY5", "QY6"]):
				identify_name = "Doctor"
			elif(identify_name in ["YN1", "YN2", "YN3", "YN4", "YN5", "YN6"]):
				identify_name = "Nurse"
			print("identify name:{}, score:{}".format(identify_name, round(1-score, 2)))
			
			#generate a new tracker for the person
			z = np.expand_dims([n for a in z for n in a], axis=0).T
			tmp_trk = Tracker() # Create a new tracker
			x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
			tmp_trk.x_state = x
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box = xx
			tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
			tmp_trk.personReID_info['personID'] = identify_name #assign the reidentified personID for the tracker
			
			#assign the tracker attribute to new tracker when loose tracking a person but re_id him
			if len(unmatched_trks)>0:
				for trk_idx in unmatched_trks:
					trk_old = tracker_list[trk_idx]
					if trk_old.personReID_info['personID'] == identify_name:
						tmp_trk.have_washed_hand = trk_old.have_washed_hand
						tmp_trk.hand_clean = trk_old.hand_clean
						tmp_trk.have_touched_pat = trk_old.have_touched_pat
						tmp_trk.violate_rule = trk_old.violate_rule

			tracker_list.append(tmp_trk)
			x_box.append(xx)

	# Deal with unmatched tracks       
	if len(unmatched_trks)>0:
		for trk_idx in unmatched_trks:
			tmp_trk = tracker_list[trk_idx]
			tmp_trk.no_losses += 1
			tmp_trk.predict_only()
			xx = tmp_trk.x_state
			xx = xx.T[0].tolist()
			xx =[xx[0], xx[2], xx[4], xx[6]]
			tmp_trk.box =xx
			x_box[trk_idx] = xx

	# The list of tracks to be annotated and draw the figure
	good_tracker_list =[]
	for trk in tracker_list:
		if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
			good_tracker_list.append(trk)
			x_cv2 = trk.box
			trackerID_str="Unknown Person:"+str(trk.id)
			if trk.personReID_info['personID'] == "Unknown":
				trk.personReID_info['personID'] = "Unknown Person:"+str(trk.id) # Change the personID for unknown person
			
			frame= draw_box_label(frame, x_cv2, personReID_info=trk.personReID_info) # Draw the bounding boxes for person
	#book keeping
	deleted_tracks = filter(lambda x: x.no_losses > max_age, tracker_list)

	#judge whether the person has washed hand before leaving and add the deleted tracker into the tracke_id_list
	for trk in deleted_tracks:
		print(trk.box, trk.hits)
		if (trk.box[2] >= 640 or trk.box[1]<0) and (trk.hits >= 10):
			if trk.have_touched_pat and (not trk.hand_clean):
				if trk.violate_rule == 2:
					trk.violate_rule = 3
				else:
					trk.violate_rule = 1

			person_tracker_info = "ctime {}, person_ID {}, sub_ID {}".format(int(time.time()), trk.personReID_info['personID'], str(trk.id))
			alarm = " washed_hand {},touched_patient {},violate_rule {}".format(str(trk.have_washed_hand),str(trk.have_touched_pat),str(trk.violate_rule))
			print(trk.personReID_info['personID']+":"+person_tracker_info+alarm)
			info = "insert into HANDEMO (PERSON, CTIME, HLOC, PLOC, HAND, PATIENT, JUDGE) \
			          values ('{}', {}, '{}', '{}', {}, {}, {})".format(trk.personReID_info['personID'], int(time.time()), '', '', \
			          	int(trk.have_washed_hand), int(trk.have_touched_pat), int(trk.violate_rule))
			cur.execute(info)
			if trk.violate_rule == 1 or trk.violate_rule == 3:
				cmd = "play After.wav"
				subprocess.Popen(cmd, shell=True)
			if trk.violate_rule == 2:
				cmd = "play Before.wav"
				subprocess.Popen(cmd, shell=True)
			#if trk.violate_rule != 0:
			#	cmd1 = "play Beep.wav"
			#	subprocess.Popen(cmd1, shell=True)


		track_id_list.append(trk.id)

	tracker_list = [x for x in tracker_list if x.no_losses<=max_age]

	#judge whether this guy has washed has hands
	#for all detected hand in sink
	if len(hand_in_sink):
		for w_h_box in hand_in_sink:
			for trk in good_tracker_list:
				if wash_hand_detector(trk,w_h_box):
					person_tracker_info = "ctime {}, person_ID {}, sub_ID {}".format(int(time.time()), trk.personReID_info['personID'] , str(trk.id))
					location_info = " hand_location {}, person_location {}".format(str(w_h_box), str(trk.box))
					alarm = " washed_hand {},touched_patient {},violate_rule {}".format(str(trk.have_washed_hand),str(trk.have_touched_pat),str(trk.violate_rule))
					#alarm = "washed_hand {},touched_patient {},hand_clean {}".format(str(trk.have_washed_hand),str(trk.have_touched_pat),str(trk.hand_clean))
					cv2.putText(frame,alarm, (w_h_box[0][0],w_h_box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,191,0), 1, cv2.LINE_AA)
					print(trk.personReID_info['personID']+":"+person_tracker_info+location_info+alarm)
					info = "insert into HANDEMO (PERSON, CTIME, HLOC, PLOC, HAND, PATIENT, JUDGE) \
			          values ('{}', {}, '{}', '{}', {}, {}, {})".format(trk.personReID_info['personID'], int(time.time()), str(w_h_box), str(trk.box), \
			          	int(trk.have_washed_hand), int(trk.have_touched_pat), int(trk.violate_rule))
					cur.execute(info)
					if trk.violate_rule == 1 or trk.violate_rule == 3:
						cmd = "play After.wav"
						subprocess.Popen(cmd, shell=True)
					if trk.violate_rule == 2:
						cmd = "play Before.wav"
						subprocess.Popen(cmd, shell=True)
					#if trk.violate_rule != 0:
					#	cmd1 = "play Beep.wav"
					#	subprocess.Popen(cmd1, shell=True) 

	#for all detected hand in patient
	if len(hand_in_patient):
		for t_p_box in hand_in_patient:
			for trk in good_tracker_list:
				if touch_patient_detector(trk,t_p_box):
					person_tracker_info = "ctime {}, person_ID {}, sub_ID {}".format(int(time.time()), trk.personReID_info['personID'], str(trk.id))
					location_info = " hand_location {}, person_location {}".format(str(t_p_box), str(trk.box))
					alarm = " washed_hand {},touched_patient {},violate_rule {}".format(str(trk.have_washed_hand),str(trk.have_touched_pat),str(trk.violate_rule))
					#alarm = "washed_hand {},touched_patient {},hand_clean {}".format(str(trk.have_washed_hand),str(trk.have_touched_pat),str(trk.hand_clean))
					cv2.putText(frame,alarm, (t_p_box[0][0],t_p_box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
					print(trk.personReID_info['personID']+":"+person_tracker_info+location_info+alarm)
					info = "insert into HANDEMO (PERSON, CTIME, HLOC, PLOC, HAND, PATIENT, JUDGE) \
			          values ('{}', {}, '{}', '{}', {}, {}, {})".format(trk.personReID_info['personID'], int(time.time()), str(t_p_box), str(trk.box), \
			          	int(trk.have_washed_hand), int(trk.have_touched_pat), int(trk.violate_rule))
					cur.execute(info)
					if trk.violate_rule == 1 or trk.violate_rule == 3:
						cmd = "play After.wav"
						subprocess.Popen(cmd, shell=True)
					if trk.violate_rule == 2:
						cmd = "play Before.wav"
						subprocess.Popen(cmd, shell=True) 
					#if trk.violate_rule != 0:
					#	cmd1 = "play Beep.wav"
					#	subprocess.Popen(cmd1, shell=True)
	return frame


def gen_frames():  # generate frame by frame from camera
	#stream detection
	#cap = open_cam_onboard(640, 480)
	#uri = "rtsp://admin:admin@192.168.1.106:554/stream2"
	uri = "rtsp://admin:edge1234@192.168.1.110:554/cam/realmonitor?channel=1&subtype=1"
	cap = open_cam_rtsp(uri, 640, 480, 200)
	im_width, im_height = (cap.get(3), cap.get(4))

	if not cap.isOpened():
		sys.exit('Failed to open camera!')

	# allow the camera to warmup
	#time.sleep(0.1)
	frame_rate_calc = 1
	#freq = cv2.getTickFrequency()
	#print(freq)\
	counter=0

	while (cap.isOpened()):
		#t1 = cv2.getTickCount()
		counter+=1
		#if counter % 12 !=0:
		#	print(counter)
		#	continue
		t1 = time.time()
		#print ("before read:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		if counter % 5 != 0:
			ret, frame = cap.read()
			#print ("after read", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
			continue

		#logger.info("FPS: {0:.2f}".format(frame_rate_calc))
		#cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
		#			cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#result = api.get_person_bbox(frame, thr=0.6)  #add functions to this line
		frame = handle_frames(frame)
		#print ("after handle", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

		t2 = time.time()
		#print("one frame takes {0:.2f}".format(t2-t1))
		frame_rate_calc = 1 / (t2 - t1)
		#if frame_rate_calc < 15:
		#	frame_rate_calc = 2*frame_rate_calc

		cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
					cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

		#if counter < 5:
		#	plt.imshow(frame[:, :, ::-1])
		#	plt.show()
		#	continue		

		# show the frame
		#cv2.imshow("Stream from EdgeNX1", frame)
		#key = cv2.waitKey(1) & 0xFF

		#t2 = cv2.getTickCount()
		
		#time1 = (t2 - t1) / freq
		#frame_rate_calc = 1 / time1
		#print("one frame takes {0:.2f}".format(t2-t1))
		
		(flag, outputFrame) = cv2.imencode(".jpg", frame)
		yield (b'--frame\r\n'
					   b'Content-Type: image/jpeg\r\n\r\n' + bytearray(outputFrame) + b'\r\n')
		

		# if the `q` key was pressed, break from the loop
		#if key == ord("q"):
		#	break
    
@app.route('/video_feed')
def video_feed():
	#Video streaming route. Put this in the src attribute of an img tag
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
	"""Video streaming home page."""
	return render_template('index.html')

tmp_time = 0
case_11 = 0
case_12 = 0
case_13 = 0
case_14 = 0
case_21 = 0
case_22 = 0
case_23 = 0
case_24 = 0


@app.route("/data")
def getdata():
	global case_11
	global case_12
	global case_13
	global case_14
	global case_21
	global case_22
	global case_23
	global case_24
	global tmp_time

	conn = sqlite3.connect('handwash.db', isolation_level=None)
	print("Query database successfully")
	cur = conn.cursor()
	
	
	if tmp_time > 0:
		sql_1 = "select * from HANDEMO where CTIME > %s and PERSON='Doctor' and HLOC=''" %(tmp_time)
		sql_2 = "select * from HANDEMO where CTIME > %s and PERSON='Nurse' and HLOC=''" %(tmp_time)
		sql_3 = "select * from HANDEMO"
	else:
		sql_1 = "select * from HANDEMO where PERSON='Doctor' and HLOC=''"
		sql_2 = "select * from HANDEMO where PERSON='Nurse' and HLOC=''"
		sql_3 = "select * from HANDEMO"

	cur.execute(sql_1)
	records_1 = cur.fetchall()
	cur.execute(sql_2)
	records_2 = cur.fetchall()
	cur.execute(sql_3)
	records_3 = cur.fetchall()
	
	if len(records_3) > 0:
		tmp_time = records_3[-1][1]
	
	for records in records_1:
		records_1_case = judge(records)
		case_11 = case_11 + records_1_case[0]
		case_12 = case_12 + records_1_case[1]
		case_13 = case_13 + records_1_case[2]
		case_14 = case_14 + records_1_case[3]
	for records in records_2:
		records_2_case = judge(records)
		case_21 = case_21 + records_2_case[0]
		case_22 = case_22 + records_2_case[1]
		case_23 = case_23 + records_2_case[2]
		case_24 = case_24 + records_2_case[3]

	

	records_1_case = [case_11, case_12, case_13, case_14]
	records_2_case = [case_21, case_22, case_23, case_24]

	records_1_case.append(sum(records_1_case)-records_1_case[0])
	records_2_case.append(sum(records_2_case)-records_2_case[0])

	results = [records_1_case, records_2_case]

	return render_template("data.html", results = results) 

	
def judge(records):
	case_1, case_2, case_3, case_4 = 0, 0, 0, 0

	if records[-1] == 0 :
		case_1+=1
	if records[-1] == 1 :
		case_2+=1
	if records[-1] == 2 :
		case_3+=1
	if records[-1] == 3 :
		case_4+=1

	case = [case_1, case_2, case_3, case_4]

	return case

if __name__ == '__main__':
	img = cv2.imread('example.jpg')
	img = handle_frames(img)
	#plt.imshow(img[:, :, ::-1])
	print("show frame")
	#plt.show()
	app.run(host='0.0.0.0', port='5000')
	#gen_frames()

