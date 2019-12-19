import cv2
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
glass_cascade= cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
ear_cascade =cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')

def detect(gray, frame):
    """ Input = greyscale image or frame from video stream
      Output = Image with rectangle box in the face
    """
    # Now get the tuples that detect the faces using above cascade  
    # faces are the tuples of 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face
    # grey means the input image to the detector
    # 1.3 is the kernel size or size of image reduced when applying the detection
    # 5 is the number of neighbors after which we accept that is a face
    # Now iterate over the faces and detect eyes
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        eye_done=0
        eyepos=0
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness
        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print(face_feat_get(frame,w,h))
        # Detect eyes now
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # Now draw rectangle over the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            print(eye_feat_get(frame,ew,eh,w,h))
            eye_done=1
            eyepos=ex+eh//2
        if eye_done==0:
            glass_eye=glass_cascade.detectMultiScale(roi_gray, 1.1, 1)
            for (gx, gy, gw, gh) in glass_eye:
                cv2.rectangle(roi_color, (gx,gy), (gx+gw, gy+gh), (255, 255, 0), 1)
                print(eye_feat_get(frame,gw,gh,w,h))
                eyepos=gx+gh//2
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (mx,my,mw,mh) in mouth:
            if my<eyepos:
                continue
            else:
                cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh), (0, 255, 255), 1)
                print(mouth_feat_get(frame,mw,mh))
        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (nx,ny,nw,nh) in mouth:
            cv2.rectangle(roi_color, (nx,ny), (nx+nw, ny+nh), (0, 0, 0), 1)
            print(nose_feat_get(frame,nw,nh))
        ear = ear_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (eax,eay,eaw,eah) in ear:
            cv2.rectangle(roi_color, (eax,eay), (eax+eaw, eay+eah), (255, 255, 255), 1)
        
    return frame
def eye_feat_get(frame,ew,eh,w,h):
    eyewperh=ew/eh
    eyelength=math.sqrt(ew^2+eh^2)
    eyelengthperface=ew/w
    if eyelength==0:
        return None
    eye_feat=('Eye',eyewperh,eyelength,eyelengthperface)
    return eye_feat
def face_feat_get(frame,w,h):
    facewperh=w/h
    facelength=math.sqrt(w^2+h^2)
    if facelength==0:
        return None
    face_feat=('Face',facewperh,facelength)
    return face_feat
def mouth_feat_get(frame,mw,mh):
    mouthwperh=mw/mh
    mouthlength=math.sqrt(mw^2+mh^2)
    if mouthlength==0:
        return None
    mouth_feat=('Mouth',mouthwperh,mouthlength)
    return mouth_feat
def nose_feat_get(frame,nw,nh):
    nosewperh=nw/nh
    noselength=math.sqrt(nw^2+nh^2)
    if noselength==0:
        return None
    nose_feat=('Nose',nosewperh,noselength)
    return nose_feat