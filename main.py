!pip install opencv-python
import cv2
import module1
import math
video_capture = cv2.VideoCapture(0)
# Run the infinite loop
while True:
    # Read each frame
    ret, frame = video_capture.read()
    #print(frame)
    # Convert frame to grey because cascading only works with greyscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Call the detect function with grey image and colored frame
    canvas = module1.detect(gray, frame)
    # Show the image in the screen
    cv2.imshow("Video", canvas)
    # Put the condition which triggers the end of program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()