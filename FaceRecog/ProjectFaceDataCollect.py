#made by mohtashim kamran
from cv2 import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

#We are going to do face detection using haarcascade
#Face Detection
face_cascade = cv2.CascadeClassifier("Haarcascade_Frontalface.xml")

skip = 0
face_data = []#face_data is an list
#We want to store tenth face in a particular location ,so create a path variable and we are going to store it in a 6)Face Recognition Project - Generating Selfie Training Data using WebCam folder .It will be an empty folder in my directory .We are going to store the frame in a grayscale image(It should be gray scale image to save memory)
dataset_path = './data/'
file_name = input("Enter the name of the person : ")
while True:
	ret,frame = cap.read()

	if ret==False:#If due to any reason frame is not captured try it again
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#cv2.COLORBGR_2GRAY is color mode#It will convert RGB frame to GRAY frame
	

	faces = face_cascade.detectMultiScale(frame,1.3,5)#This faces will be a list and each face is a tupule
	if len(faces)==0:
		continue
		
	faces = sorted(faces,key=lambda f:f[2]*f[3])#f is faces here [(x,y,w,h),(,,,).....]#indexing is from 0 so we will multiply f[2]*f[3] i.e,w*h
    #We are going to make a bounding box around each faces 
    #So we are going to iterate over faces
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))#Number of rows should be same as number of faces,Number of columns will be figured out automatically
print(face_data.shape)#(7,30000)because we have chosen frame(RGB) 3 for RGB has been multiplied here,if we have chosen gray_frame then it will be(7,10000)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()
