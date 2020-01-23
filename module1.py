import cv2
import math

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
glass_cascade= cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
lear_cascade =cv2.CascadeClassifier('haarcascade_mcs_leftear.xml')

def detect(gray, frame):
    """ Input = greyscale image or frame from video stream
      Output = Image with rectangle box in the face
    """
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        facepos=[x+w//2,y+h//2]
        eye_done=0
        eyepos=0
        nose_done=0
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 1)
        cv2.line(frame, (x,y), (x+w, y+h), (255,0,0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        face_feat=feat_get(frame,w,h))
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            if eye_done==2:
                break
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            eye_feat=feat_get(frame,ew,eh,w,h))
            eye_done+=1
            eyepos=ey+eh//2
        if eye_done!=2:
            glass_eye=glass_cascade.detectMultiScale(roi_gray, 1.1, 1)
            for (gx, gy, gw, gh) in glass_eye:
                cv2.rectangle(roi_color, (gx,gy), (gx+gw, gy+gh), (255, 255, 0), 1)
                eye_feat=feat_get(frame,gw,gh,w,h))
                eyepos=gy+gh//2
                eye_done+=1
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (mx,my,mw,mh) in mouth:
            mouthpos=my+mh//2
            if mouthpos<eyepos or mouthpos<y+h//2:
                continue
            else:
                cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh), (0, 255, 255), 1)
                mouth_feat=feat_get(frame,mw,mh))
        nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (nx,ny,nw,nh) in nose:
            if nose_done==0:
                nosepos=nx+nw//2
                if nosepos-facepos[0]>nw:
                    continue
                cv2.rectangle(roi_color, (nx,ny), (nx+nw, ny+nh), (128, 128, 0), 1)
                nose_feat=feat_get(frame,nw,nh))
                nose_done+=1
            else: 
                break
        lear = lear_cascade.detectMultiScale(roi_gray, 1.1, 1)
        for (leax,leay,leaw,leah) in lear:
            cv2.rectangle(roi_color, (leax,leay), (leax+leaw, leay+leah), (255, 255, 255), 1)
        
    return frame

def feat_get(pw,ph,w,h):
    partwperh=pw/ph
    partlengthperface=[pw/w,pw/h]
    partareaperface=pw*ph/w/h
    if pw==0 or ph==0:
        return None
    part_feat=[partwperh,partlengthperface,partareaperface]