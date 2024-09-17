# Import necessary libraries
import cv2
import time

# Main code
def main():
    # Initialize OpenCV and the face recognizer
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")

    name_list = ["", "Amira Sayed", "Amira"]  # Modify this list based on the number of people you want to recognize

    while True:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            serial_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            
            if conf > 50:
                # Recognized person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name_list[serial_id], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Unknown person
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show the live webcam feed with face detection/recognition
        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break

    # Cleanup
    video.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == "__main__":
    main()
