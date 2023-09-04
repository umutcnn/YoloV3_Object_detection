"""
https://pjreddie.com/darknet/yolo/ sitesine giriyorum ordan 
YOLOv3-416 modelinin cfg ve Weights dosyalarını indiriyorum

-->> YOLOv3 ile Görüntü Üzerinde Nesne Algılama

"""

#kütüphaneleri ice aliyorum
import cv2
import numpy as np

image = cv2.imread("C:/Users/UMUT/python/bilgisayarli_goru_yolov4_nesne_tanima/YOLOv3_ile_nesne_tanima/img/people.jpg")
#print(img)

image_width = image.shape[1]
image_height = image.shape[0]
#print(img_width)
#print(img_height)

#1/255 goruntunun piksel degerini 255 e boler bu bir cesit normalizasyon islemidir literatürde boyle gecer
#(416,416) modelimizin girdisi olan gorselin buyuklugu
#swapRB=True RGB'den BGR'ye donustureyim mi
#crop= goruntuyu kırpayım mı
image_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB=True, crop= False)

labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

#5 tane renk olusturuyoruz
colors = ["0,255,255","0,0,255","255,0,0","255,255,0","0,255,0"]

#bu renkleri int ceviriyoruz
colors = [np.array(color.split(",")).astype("int") for color in colors ]

#colorslarin tek bir arrayda olmasini istiyorum
colors = np.array(colors)

#ayni degerleri alt alta ekliyorum
colors = np.tile(colors,(18,1))

#modellerimizi aliyoruz
model = cv2.dnn.readNetFromDarknet(r'C:/Users/UMUT/python/bilgisayarli_goru_yolov4_nesne_tanima/YOLOv3_ile_nesne_tanima/model/yolov3.cfg' , r'C:/Users/UMUT/python/bilgisayarli_goru_yolov4_nesne_tanima/YOLOv3_ile_nesne_tanima/model/yolov3.weights')

#lgılama islemini yapabilmek icin modelimizin katman isimlerini aliyorum
model_layers = model.getLayerNames()

#ciktilarimizi aliyoruz
output_layer = [model_layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

#resmi modele vermek icin blob yazmistik
model.setInput(image_blob)

#ciktilarimizi forward sokrark tespit etmis oluyoruz
detection_layers = model.forward(output_layer)

# Creating a couple of lists for Non-maximum suppression
ids_list = []
boxes_list = []
confidences_list = []

# ciktilari islemek icin bir kac dongu kuruyoruz 
#tum nesne tespit katmanlarini dolasiyorum
for detection_layer in detection_layers:
    #nesne tespitini yani kutucugu ve etiketi buluyorum
    for object_detection in detection_layer:

        #ilk bes ogeyi aliyorum yuksek dogrulukta olmasi icin
        scores = object_detection[5:]

        #max score al
        predicted_id = np.argmax(scores)

        #guven skoru
        confidence = scores[predicted_id]

        #guvenlik skoru %20 den buyukse
        if confidence > 0.20:
            #hangi lavel oldugunu buluyoruz
            label = labels[predicted_id]

            #sinirlayici kutunun kordinatlarini buluyruz
            bounding_box = object_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
            (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype('int')

            #merkezini buluyoruz
            start_x = int(box_center_x - (box_width / 2))
            start_y = int(box_center_y - (box_height / 2))

            #ids_list tahmin edilen sınıfların indislerini temsil eder
            #confidences_list guven değerlerini
            #boxes_list sınırlayıcı kutuların koordinatlarını icerir
            ids_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

# Non-maximum suppression
max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

for max_id in max_ids:
    max_class_id = max_id
    box = boxes_list[max_class_id]

    #x,y,w,h degerlerini buluyorum
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]

    #etiketini buluyorum
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidences_list[max_class_id]


    end_x = start_x + box_width
    end_y = start_y + box_height

    #renk
    box_color = colors[predicted_id]
    box_color = [int(each) for each in box_color]

    label = '{}: {:.2f}%'.format(label, confidence * 100)
    print('Predicted Object {}'.format(label))

    #cizim islemleri
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), box_color, 1)
    cv2.putText(image, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

cv2.imshow('Detection Window', image)
cv2.waitKey(0)
