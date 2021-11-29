import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort import preprocessing1, nn_matching1
from deep_sort.detection import Detection
from deep_sort.detection1 import Detection1
from deep_sort.tracker import Tracker
from deep_sort.tracker1 import Tracker1
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video1', './data/video/test1.mp4', 'path to input video1 or set to 1 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    metric1 = nn_matching1.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    tracker1 = Tracker1(metric1)
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    input_size1 = FLAGS.size
    video_path = FLAGS.video
    video_path1 = FLAGS.video1
    
    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
        interpreter1 = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter1.allocate_tensors()
        input_details1 = interpreter1.get_input_details()
        output_details1 = interpreter1.get_output_details()
        print(input_details1)
        print(output_details1)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
        vid1 = cv2.VideoCapture(int(video_path1))
    except:
        vid = cv2.VideoCapture(video_path)
        vid1 = cv2.VideoCapture(video_path1)
    out = None
    out1 = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    
        width1 = int(vid1.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(vid1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps1 = int(vid1.get(cv2.CAP_PROP_FPS))
        codec1 = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out1 = cv2.VideoWriter(FLAGS.output, codec1, fps1, (width1, height1))


    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        return_value1, frame1 = vid1.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (854,480))
            image = Image.fromarray(frame)
        if return_value1:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, (854,480))
            image1 = Image.fromarray(frame1)    
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()
        image_data1 = cv2.resize(frame1, (input_size1, input_size1))
        image_data1 = image_data1 / 255.
        image_data1 = image_data1[np.newaxis, ...].astype(np.float32)
        ##start_time1 = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            interpreter1.set_tensor(input_details1[0]['index'], image_data1)
            interpreter1.invoke()
            pred1 = [interpreter1.get_tensor(output_details1[i]['index']) for i in range(len(output_details1))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
                boxes1, pred_conf1 = filter_boxes(pred1[1], pred1[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size1, input_size1]))                           
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
                boxes1, pred_conf1 = filter_boxes(pred1[0], pred1[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size1, input_size1]))                          
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            batch_data1 = tf.constant(image_data1)
            pred_bbox1 = infer(batch_data1)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]
            for key, value1 in pred_bbox1.items():
                boxes1 = value1[:, :, 0:4]
                pred_conf1 = value1[:, :, 4:] 

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=25,
            max_total_size=25,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )
        boxes1, scores1, classes1, valid_detections1 = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes1, (tf.shape(boxes1)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf1, (tf.shape(pred_conf1)[0], -1, tf.shape(pred_conf1)[-1])),
            max_output_size_per_class=25,
            max_total_size=25,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )  
        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        num_objects1 = valid_detections1.numpy()[0]
        bboxes1 = boxes1.numpy()[0]
        bboxes1 = bboxes1[0:int(num_objects1)]
        scores1 = scores1.numpy()[0]
        scores1 = scores1[0:int(num_objects1)]
        classes1 = classes1.numpy()[0]
        classes1 = classes1[0:int(num_objects1)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)
        original_h1, original_w1, _ = frame1.shape
        bboxes1 = utils.format_boxes(bboxes1, original_h1, original_w1)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        pred_bbox1 = [bboxes1, scores1, classes1, num_objects1]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        class_names1 = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        #allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        #allowed_classes = ['person','cup','cellphone','car','bus','truck','aeroplane']

        allowed_classes = ['car','bus','aeroplane']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        names1 = []
        deleted_indx = []
        deleted_indx1 = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)  
            else:
                names.append(class_name)
               
        for j in range(num_objects1): 
            class_indx1 = int(classes1[j])
            class_name1 = class_names1[class_indx1]  
            if class_name1 not in allowed_classes:
                deleted_indx1.append(j)   
            else:
                names1.append(class_name1)    
        
        names = np.array(names)
        count = len(names)
        names1 = np.array(names1)
        count1 = len(names1)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
            cv2.putText(frame1, "Objects being tracked: {}".format(count1), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count1))

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        bboxes1 = np.delete(bboxes1, deleted_indx1, axis=0)
        scores1 = np.delete(scores1, deleted_indx1, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        features1 = encoder(frame1, bboxes1)
        detections1 = [Detection1(bbox1, score1, class_name1, feature1) for bbox1, score1, class_name1, feature1 in zip(bboxes1, scores1, names1, features1)]
        
        # extracting the features vector for comparison and re identification
        meanfeatures = np.ndarray.mean(features)
        meanfeatures1 = np.ndarray.mean(features1)      

        # calculating correlation and setting the values for printing
        R = np.around(np.multiply(np.corrcoef(features,features1),100), decimals=0)
        tracklist, tracklist1 = [], []

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        colors1 = [cmap(j)[:3] for j in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]    

        boxs1 = np.array([d.tlwh1 for d in detections1])
        scores1 = np.array([d.confidence1 for d in detections1])
        classes1 = np.array([d.class_name1 for d in detections1])
        indices1 = preprocessing1.non_max_suppression(boxs1, classes1, nms_max_overlap, scores1)
        detections1 = [detections1[i] for i in indices1]   
 
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        tracker1.predict1()
        tracker1.update1(detections1)
        
        # update tracks
        for i in indices: 
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                # draw bbox, put the correlation values and track same objects on the screen
                bbox = track.to_tlbr()
                class_name = track.get_class()
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                tracklist.append(int(track.track_id))
                
                if np.allclose(meanfeatures, meanfeatures1, atol=0.0015, rtol=1) == True and R[0,i+1] > 50:
                    T = 'Same'
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id))+len("Same       "))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name +  "-" + str(track.track_id) + ": " + T + " " + str(R[0,1]),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)  
                    if len(indices) > 2:
                        cv2.putText(frame, class_name + "-" + str(tracklist[i]) + ": " + str(R[0,i+1]) + "%" , (5, i*17+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,215,0), 2) 

                else:
                    T = ''   
                    
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        for i in indices1:
            for track1 in tracker1.tracks1:
                if not track1.is_confirmed1() or track1.time_since_update > 1:
                    continue 
                
                # draw bbox, put the correlation values and track same objects on the screen
                bbox1 = track1.to_tlbr1()
                class_name1 = track1.get_class1()
                color1 = colors1[int(track1.track_id1) % len(colors1)]
                color1 = [j * 255 for j in color1]
                cv2.rectangle(frame1, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), color1, 2)
                tracklist1.append(int(track1.track_id1))

                if np.allclose(meanfeatures, meanfeatures1, atol=0.0015, rtol=1) == True and R[0,i+1] > 50:
                    T = 'Same'
                    cv2.rectangle(frame1, (int(bbox1[0]), int(bbox1[1]-30)), (int(bbox1[0])+(len(class_name1)+len(str(track1.track_id1))+len("Same      "))*17, int(bbox1[1])), color1, -1)
                    cv2.putText(frame1, class_name1 + "-" + str(track1.track_id1) + ":" + T + " " + str(R[0,1]),(int(bbox1[0]), int(bbox1[1]-10)),0, 0.75, (255,255,255),2)        
                    if len(indices1) > 2:
                        cv2.putText(frame1, class_name1 + "-" + str(tracklist1[i]) + ": " + str(R[0,i+1]) + "%" , (5, i*17+20), cv2.FONT_HERSHEY_PLAIN, 1, (0,215,0), 2)   
                                    
                else:
                    T = ''   
                
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track1.track_id1), class_name1, (int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3]))))
        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result1 = np.asarray(frame1)
        result1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
     
        if not FLAGS.dont_show:
            cv2.imshow("Output Video 1", result)  
            cv2.imshow("Output Video 2", result1)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
            out1.write(result1)       
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    vid.release()
    vid1.release()    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
