import cv2
import os
import numpy as np

# Configuration files
cfg_file = 'znaki.cfg' # Change it for your .cfg file
weights_file = 'znaki_best.weights' #C hange for your weights
names_file = 'znaki.names' # Change for your class names

net = cv2.dnn.readNet(weights_file, cfg_file)
classes = None

with open(names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Path to folder with photos to analyze
folder_path = 'Zdjecia'

# All photos in the folder are taken into account, as long as they have proper sufix
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        original_height, original_width = image.shape[:2]

        # Prepering photo for processing with YOLOv4
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Getting detection
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Data after processing with YOLOv4
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Detection threshold(can be changed)
                    center_x = int(detection[0] * original_width)
                    center_y = int(detection[1] * original_height)
                    w = int(detection[2] * original_width)
                    h = int(detection[3] * original_height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Taking into account only biggest detection
        max_area = 0
        max_area_index = -1

        for i in range(len(boxes)):
            box = boxes[i]
            x, y, w, h = box
            area = w * h 

            if area > max_area:
                max_area = area
                max_area_index = i

        # Checking if biggest detection was found
        if max_area_index != -1:
            box = boxes[max_area_index]
            x, y, w, h = box
            label = f"{classes[class_ids[max_area_index]]}"

            # Scaling photo to 1/4 of it's original size
            scaled_image = cv2.resize(image, (original_width // 4, original_height // 4))

            # Drawing detection frame onto photo
            scaled_x = int(x / 4)  
            scaled_y = int(y / 4)
            scaled_w = int(w / 4)
            scaled_h = int(h / 4)
            cv2.rectangle(scaled_image, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), 4)

            # Calculating centre of detection frame and drawing circle on proper position
            scaled_center_x = scaled_x + scaled_w // 2
            scaled_center_y = scaled_y + scaled_h // 2
            nscaled_center_x = 4*scaled_center_x
            nscaled_center_y = 4*scaled_center_y
            cv2.circle(scaled_image, (scaled_center_x, scaled_center_y), 3, (0, 255, 0), -1)

            # Information about coordinates of original photo and smaller one
            print(f"Sign was detected: {label}, Center coordinations: ({scaled_center_x}, {scaled_center_y})")
            print(f"Coordinations on normal photo: ({nscaled_center_x}, {nscaled_center_y})")

            cv2.putText(scaled_image, label, (scaled_x, scaled_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Showing scaled photo
            cv2.imshow('Object Detection', scaled_image)
            key = cv2.waitKey(0)

            if key == 27:  # Press ESC to exit the program
                cv2.destroyAllWindows()