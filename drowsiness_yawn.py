#python drowniness_yawn.py --webcam webcam_index

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
# import os

import pyttsx3

BlinkCounter = 0
Wb=0
Wr=0
Wy=0
Wt=0
#checks the number of times blinks every minute
def minute_passed():
    global Wb
    global BlinkCounter
    while True:
        time.sleep(60)
        print("BlinkCounter after one:",BlinkCounter)
        if(BlinkCounter<5 or BlinkCounter>25):
            Wb=1
        print("one passed")
        BlinkCounter=0
    
#For voice alert and vibrator
def alarm(msg):
    global BlinkCounter
    global alarm_status
    global alarm_status2
    global saying
    conseq = BlinkCounter
    converter = pyttsx3.init()
    converter.setProperty('rate', 150)
    converter.setProperty('volume', 0.5)
    #any two values of Wb Wr Wt Wy is active
    if "Special DROWSINESS" in msg:
        print("call vibrator")
        converter.setProperty('volume', 0.9)
    #alarm_status=eye closed
    while alarm_status:
        # print('call')
        # s = 'espeak "'+msg+'"'
        # os.system(s)
        converter.say(msg)
        BlinkCounter += 1
        #when eye is closed for too long
        if(conseq+1<BlinkCounter):
            Wt=1
        converter.runAndWait()
        converter.stop()
    #alarm_status2=yawning
    if alarm_status2:
        # print('call')
        saying = True

        # s = 'espeak "' + msg + '"'
        # os.system(s)
        converter.say(msg)
        saying = False
        converter.runAndWait()
        converter.stop()
#eye_aspect_ratio for each call
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear
# eye_aspect_ratio
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance
f=0
def shadeMoter():
    global f
    print("down")
    #motor down
    time.sleep(10*60.0)
    #motor up
    print("up")
    f=0
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

LUMINACANCE_THRESH = 190
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 24
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

print("-> Loading the predictor and detector...")
# detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

t = Thread(target=minute_passed)
t.deamon = True
t.start()

print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
# vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

while True:
    print("BlinkCounter :",BlinkCounter)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        

        #Luminacance
        avg=0
        r,c=0,0
        fwidth=w//4
        fheight=h//4
        rect0010=int(x+fwidth)
        rect0111=int(x+2*fwidth)
        rect1011=int(y+2*fheight)
        rect0001=int(y+fheight)
        for row in gray[rect0001:rect1011]:
        	avg+=sum(row[rect0010:rect0111])
        	c=len(row[rect0010:rect0111])
        r=len(gray[rect0001:rect1011])
        avg=avg/(r*c)
        print("Luminacance : ",avg)
        if(avg>LUMINACANCE_THRESH and f==0):
            f=1
            t = Thread(target=shadeMoter)
            t.deamon = True
            t.start()
        # elif(avg):


        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES*2:
                if alarm_status == False:
                    alarm_status = True

                    Wr=1

                    t = Thread(target=alarm, args=('wake up sir',))
                    t.deamon = True
                    t.start()

                # cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    Wy=1
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
        else:
            alarm_status2 = False
        

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    print("Wr : ",Wr)
    print("Wy : ",Wy)
    print("Wb : ",Wb)
    print("Wt : ",Wt)
    print("Wr+Wy+Wb+Wt : ",Wr+Wy+Wb+Wt)
    if((Wr+Wy+Wb+Wt)>=2):
        Wb=0
        Wr=0
        Wy=0
        Wt=0
        print("Special DROWSINESS Special DROWSINESS Special DROWSINESS")
        t = Thread(target=alarm, args=('Special DROWSINESS Special DROWSINESS Special DROWSINESS',))
        t.deamon = True
        t.start()
        cv2.putText(frame, "Special DROWSINESS: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
