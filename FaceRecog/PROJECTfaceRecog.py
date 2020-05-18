from cv2 import cv2
import numpy as np
import os

###########KNN CODE##############
def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
    dist=[]
    for i in range(train.shape[0]):
        #get vector and label
        ix = train[i,:-1]
        iy = train[i,-1]
        #compute distance from starting point
        d=distance(test,ix)
        dist.append([d,iy])
    #sort based on distance from test point
    dk = sorted(dist,key=lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:,-1]
    #get frequencies of each label
    output = np.unique(labels,return_counts=True)
    #find max frequency in corresponding label
    index = np.argmax(output[1])
    return output[0][index]
#####################################
#Initialise Camera
cap = cv2.VideoCapture(0)

#We are going to do face detection using haarcascade
#Face Detection
face_cascade = cv2.CascadeClassifier("Haarcascade_Frontalface.xml")

skip = 0
#We want to store tenth face in a particular location ,so create a path variable and we are going to store it in a 6)Face Recognition Project - Generating Selfie Training Data using WebCam folder .It will be an empty folder in my directory .We are going to store the frame in a grayscale image(It should be gray scale image to save memory)
dataset_path = './data/'
face_data = []#it will get the x value of our data

labels =[]#it will get the y value of our data

class_id = 0 #first file that i'll load in my system will have id =0 and the 1 ,2, and so on(labels for the file)

name ={}#mapping between id and name

###################data preparation###############
for fx in os.listdir(dataset_path):
    if fx.endswith('npy'):
        #create a mapping between class id and name
        name[class_id]= fx[:-4]
        print("Loaded "+fx)

        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #create labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1 
        labels.append(target)
face_dataset = np.concatenate(face_data,axis=0) #for x train
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))  #for y train
print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing 

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		#Predicted Label (out)
		out = knn(trainset,face_section.flatten())

		#Display on the screen the name and rectangle around it
		pred_name = name[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,265,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
