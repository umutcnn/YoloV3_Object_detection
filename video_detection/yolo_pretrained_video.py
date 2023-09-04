"""
https://pjreddie.com/darknet/yolo/ sitesine giriyorum ordan 
YOLOv3-416 modelinin cfg ve Weights dosyalarını indiriyorum

-->> YOLOv3 ile Görüntü Üzerinde Nesne Algılama

"""

#kütüphaneleri ice aliyorum
import cv2
import numpy as np

#videoyu al
cap = cv2.VideoCapture(r'C:/Users/UMUT/python/bilgisayarli_goru_yolov4_nesne_tanima/7_yolov_3_nesne_tanima_video-webcam/video/people.mp4')


while True:
    # framelerimizi okuyoruz
    ret, frame = cap.read()
    

    # frame genislik ve yuksekligini aliyoruz
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # blob formatina ceviriyoruz
    #1/255 goruntunun piksel degerini 255 e boler bu bir cesit normalizasyon islemidir literatürde boyle gecer
    #(416,416) modelimizin girdisi olan gorselin buyuklugu
    #swapRB=True RGB'den BGR'ye donustureyim mi
    #crop= goruntuyu kırpayım mı
    frame_blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB = True, crop = False)

    labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
             "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
             "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
             "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
             "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
             "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
             "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
             "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
             "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
             "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    #5 tane renk olusturuyoruz
    colors = ['0, 255, 255', '0, 0, 255', '255, 0, 0', '255, 255,0', '0, 255, 0']

    #bu renkleri int ceviriyoruz
    colors = [np.array(color.split(",")).astype("int") for color in colors ]

    #colorslarin tek bir arrayda olmasini istiyorum
    colors = np.array(colors)

    #ayni degerleri alt alta ekliyorum
    colors = np.tile(colors,(18,1))

    #modellerimizi aliyoruz
    model = cv2.dnn.readNetFromDarknet(r'C:\Users\UMUT\python\bilgisayarli_goru_yolov4_nesne_tanima\5-6_YOLOv3_ile_nesne_tanima\model\yolov3.cfg' , r'C:\Users\UMUT\python\bilgisayarli_goru_yolov4_nesne_tanima\5-6_YOLOv3_ile_nesne_tanima\model\yolov3.weights')
    
    #algılama islemini yapabilmek icin modelimizin katman isimlerini aliyorum
    model_layers = model.getLayerNames()

    #ciktilarimizi aliyoruz
    output_layer = [model_layers[layer - 1] for layer in model.getUnconnectedOutLayers()]

    #videoyu modele vermek icin blob yazmistik
    model.setInput(frame_blob)

    #ciktilarimizi forward sokrark tespit etmis oluyoruz
    detection_layers = model.forward(output_layer)

    # Creating lists for non-maximum suppression
    ids = []
    boxes = []
    confidences = []

    # ciktilari islemek icin bir kac dongu kuruyoruz 
    #tum nesne tespit katmanlarini dolasiyorum
    for detection_layer in detection_layers:
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
                bounding_box = object_detection[0:4] * np.array([frame_width, frame_height, frame_width, frame_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype('int')

                #merkezini buluyoruz
                start_x, start_y = int(box_center_x - (box_width / 2)), int(box_center_y - (box_height / 2))

                #ids_list tahmin edilen sınıfların indislerini temsil eder
                #confidences_list guven değerlerini
                #boxes_list sınırlayıcı kutuların koordinatlarını icerir
                ids.append(predicted_id)
                confidences.append(float(confidence))
                boxes.append([start_x, start_y, int(box_width), int(box_height)])
    
    # Non-maximum suppression
    max_ids = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    for max_id in max_ids:

        max_class_id = max_id
        box = boxes[max_class_id]

        start_x, start_y, box_width, box_height = box[0], box[1], box[2], box[3]

        #etiketini buluyorum
        predicted_id = ids[max_class_id]
        label = labels[predicted_id]
        confidence = confidences[max_class_id]

        end_x, end_y = start_x + box_width, start_y + box_height

        #renkleri buluyorum
        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]

        label = '{}: {:.2f}%'.format(label, confidence * 100)
        print('Predicted Object {}'.format(label))

        #cizim islemleri
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 1)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Show the video
    cv2.imshow('Detection Window', frame)

    # Keep the video open till we want to close it, if we need to close it then we'll use q from the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()