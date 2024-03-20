from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    cap.set(cv2.CAP_PROP_FPS, 30)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter('output.avi', fourcc, 10, (frame_width, frame_height))

    model=YOLO("runs\\detect\\yolov8m_v8_50e\\weights\\best.pt")
    classNames = ['weapon']
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        # print('results :',results)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                print("cls :",cls)
                if cls < len(classNames) :
                    class_name=classNames[cls]
                    print("classname block")
                    label=f'{class_name}{conf}'
                else  :
                    label = f'Unknown'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                # if class_name == 'weapon':
                #     color=(0, 204, 255)
                # else:
                #     color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), (0, 204, 255),3)
                    cv2.rectangle(img, (x1,y1), c2, (0, 204, 255), -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
                print("block is working")

        yield img
        out.write(img)
        cv2.imshow("image", img)
        if cv2.waitKey(30) & 0xFF==ord('q'):
            break
    out.release()
cv2.destroyAllWindows()