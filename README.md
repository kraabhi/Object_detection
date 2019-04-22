# Basics of Object_detection
# Detect objects from image and save each object separately using imageai

# import object detection from imageai library

from imageai.Detection import ObjectDetection
import os
execution_path = os.getcwd()

# We create a new instance of the ObjectDetection class and set the model type to retinanet.Also we set the model path RetinaNet model file we downloaded and copied to the python file folder and load the model
# RetinaNet is appropriate for high-performance and high-accuracy demanding detection tasks
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

# We provide input image path where original image is placed and out put path as the new file . We set the minimum percentage probability
# so that those objects with that minimum probabilities will be detected , also we add one more argument extract_detected_objects which will extract all the objects with probability greater then or equal to 30. By default it is false , so we make it true.Object path will contain the address of all the objects detected and extracted and will be saved in the new file created above.

detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 
                          "car.jpg"), output_image_path=os.path.join(execution_path , "car_new1.jpg"),
                           minimum_percentage_probability=30,  extract_detected_objects=True)

for eachObject, eachObjectPath in zip(detections, objects_path):
print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
print("Object's image saved in " + eachObjectPath)
print("--------------------------------")

# Display the result with bounding boxes
from IPython.display import Image
Image("car_new1.jpg")

# The object detection model (RetinaNet) supported by ImageAI can detect 80 different types of objects. 
# person,   bicycle,   car,   motorcycle,   airplane,bus,   train,   truck,   boat,   traffic light,   fire hydrant,   stop_sign, parking meter,   bench,   bird,   cat,   dog,   horse,   sheep,   cow,   elephant,   bear,   zebra,  giraffe,   backpack,   umbrella,   handbag,   tie,   suitcase,   frisbee,   skis,   snowboard,sports ball,   kite,   baseball bat,   baseball glove,   skateboard,   surfboard,   tennis racket,bottle,   wine glass,   cup,   fork,   knife,   spoon,   bowl,   banana,   apple,   sandwich,   orange, broccoli,   carrot,   hot dog,   pizza,   donot,   cake,   chair,   couch,   potted plant,   bed,dining table,   toilet,   tv,   laptop,  mouse,   remote,   keyboard,   cell phone,   microwave,oven,   toaster,   sink,   refrigerator,   book,   clock,   vase,   scissors,   teddy bear,   hair dryer, toothbrush.


# We can select the objects to be detected by defining the custom objects and we just need to add one more argument in the detector.detectCustumObjectsFromImage

custom_objects = detector.CustomObjects(car=True, motorcycle=True)
detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image3.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=30)

