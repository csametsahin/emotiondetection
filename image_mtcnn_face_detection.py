import cv2
import time
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
from keras.utils import  img_to_array
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
print(cv2.__version__)

baslangic_zamani=time.time()


#tespit için görüntüyü yükleme
image_to_detect=plt.imread('20221031_1c373_fakulteeed-14c.jpg')
#load the model and load the weights
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
#declare the emotions label
emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotions_color = [(0,0,255),(0,0,255),(0,0,255),(0,255,0),(0,0,255),(0,255,0),(255,255,0)]
#nesne oluşturma
mtcnn_detector=MTCNN()
#konumları tespit etme
all_face_locations = mtcnn_detector.detect_faces(image_to_detect)
print('{} tane yüz bulunmaktadır'.format(len(all_face_locations)))

image_to_detect=cv2.cvtColor(image_to_detect,cv2.COLOR_BGR2RGB)
toplam_verimlilik=0
for index,current_face_location in enumerate(all_face_locations):
    score = current_face_location["confidence"]
    toplam_verimlilik=toplam_verimlilik+score
    x,y,width,height =current_face_location['box']
    left_x,left_y=x,y
    right_x,right_y=x+width,y+height
    
    print('Bulunan yüz {} left_x:{},left_y:{},right_x:{},right_y:{}'.format(index+1,left_x,left_y,right_x,right_y
                                                                            ))
    current_face_image = image_to_detect[left_y:right_y,left_x:right_x]
    #cv2.imshow("{}"+str(index+1),current_face_image)
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
    #resize to 48x48 px size
    current_face_image = cv2.resize(current_face_image, (48, 48))
     #convert the PIL image into a 3d numpy array
    img_pixels = img_to_array(current_face_image)
    #expand the shape of an array into single row multiple columns
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    #pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
    img_pixels /= 255 
    
    #do prodiction using model, get the prediction values for all 7 expressions
    exp_predictions = face_exp_model.predict(img_pixels) 
    #find max indexed prediction value (0 till 7)
    max_index = np.argmax(exp_predictions[0])
    #get corresponding lable from emotions_label
    emotion_label = emotions_label[max_index]
    emotion_color = emotions_color[max_index]
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.rectangle(image_to_detect,(left_x,left_y),(right_x,right_y),emotion_color,2)
    cv2.putText(image_to_detect, emotion_label, (left_x,left_y), font, 0.5, (255,255,255),1)

print(time.time() - baslangic_zamani," saniyede tamamlanmaktadır")
print("Verimillik:",(toplam_verimlilik/len(all_face_locations)*100))
cv2.imshow("Yuz Tanima",image_to_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()