
import  mediapipe as mp

import cv2 
import numpy as np 

cap = cv2.VideoCapture(0)

name = input("Enter the name of the data : ")

holistic = mp.solutions.holistic #to capture all body moments,object
hands = mp.solutions.hands #hands for showing visuals
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils #to show visuals

X = [] 
data_size = 0 #data size is  trigger

while True:
    lst = [] #will consist collection of all the rows (1020 columns of landmarks)
    
    _, frm = cap.read()
    
    frm = cv2.flip(frm, 1) #no mirror image flipcode -1(left to riht)
    
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)) 
    
    if res.face_landmarks: #if thier is someone in frame
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x) #iterating thorugh all of the objwects
            lst.append(i.y - res.face_landmarks.landmark[1].y)
            
        if res.left_hand_landmarks: #storing it on left hand side, iterating through x axis and y axis
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x) # 8th landmaek point of left hand w.r.t to thhat
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
                
        else: #if left hand is not in frame
            for i in range(42):
                lst.append(0.0) #storing it in frame
                
        if res.right_hand_landmarks: #for right hand landmark
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y) 
                #as their will be different positions in frame having different values w.rt to particular 
        else:
            for i in range(42):
                lst.append(0.0)
                
                
        X.append(lst) 
        data_size = data_size+1 #incrementing data size storing one row in data set


    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS) #used to draw landmarks
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2) #data size converted in strong Text string to be drawn.
    #it denotes the font type.
    cv2.imshow("window", frm)
    

    if cv2.waitKey(1) == 27 or data_size>99:
        cv2.destroyAllWindows()
        cap.release()
        break


np.save(f"{name}.npy", np.array(X)) #saving data in numpy format for one imogi using foramat specifier
print(np.array(X).shape)
#data shape 100 rows and 1020 columns