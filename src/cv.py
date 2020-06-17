from utils.datasets import *
from utils.utils import *
import math
from sound import run_sound
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang #ang + 360 if ang < 0 else ang

#import streamlit as st
#st.header("YoloV5 in real time on CPU")

weights = 'src/weights/yolov5s.pt'
source = "0" #'inference/images'
out = 'inference/output'
imgsz = 640
view_img = True
save_txt = True
save_img = False
webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
device = "cpu"
augment = True
conf_thres = 0.4
iou_thres = 0.5
classes = None
agnostic_nms = True
fourcc = 'mp4v'

with torch.no_grad():

    # Initialize
    device = torch_utils.select_device(device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    #image_placeholder = st.empty()

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   fast=True, classes=classes, agnostic=agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # Center line
            width, height = im0.shape[1], im0.shape[0]
            x1, y1 = width//2, height
            x2, y2 = width//2, 0
            line_thickness = 2

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #print(n)
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        centroid_x, centroid_y = int(width*(xywh[0])), int(height*(xywh[1]))
                        cv2.line(im0, (x1, y1), (centroid_x, centroid_y), (0, 255, 0), thickness=line_thickness)
                        cv2.circle(im0, (centroid_x, centroid_y), 20, (0, 255, 0))
                        #print(getAngle((centroid_x, centroid_y), (x1, y1), (x2, y2)))
                        cv2.putText(im0, str(np.round(getAngle((centroid_x, centroid_y), (x1, y1), (x2, y2)))), (centroid_x, centroid_y), 0, 1, [225, 255, 255])

                        label = '%s %.2f' % (names[int(cls)], conf)
                        if names[int(cls)] == "cell phone":
                            run_sound("cellphone.wav", np.round(getAngle((centroid_x, centroid_y), (x1, y1), (x2, y2))))
                        elif names[int(cls)] == "person":
                            run_sound("person.wav", np.round(getAngle((centroid_x, centroid_y), (x1, y1), (x2, y2))))
                        elif names[int(cls)] == "clock":
                            run_sound("clock.wav", np.round(getAngle((centroid_x, centroid_y), (x1, y1), (x2, y2))))

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            #if view_img:

            #centroid_x, centroid_y = int(width*(xywh[0])), int(height*(xywh[1]))
            
            # Line to object
            #cv2.line(im0, (x1, y1), (centroid_x, centroid_y), (0, 255, 0), thickness=line_thickness)

            #print(im0.shape)

            cv2.line(im0, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)
            
            cv2.imshow(p, im0)

            #image_placeholder.image(im0, channels="BGR")
                #time.sleep(1)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

            # Save results (image with detections)
            #if save_img:
                #if dataset.mode == 'images':
                    #cv2.imwrite(save_path, im0)
                #else:
                    #if vid_path != save_path:  # new video
                        #vid_path = save_path
                        #if isinstance(vid_writer, cv2.VideoWriter):
                            #vid_writer.release()  # release previous video writer

                        #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    #vid_writer.write(im0)

    #if save_txt or save_img:
        #print('Results saved to %s' % os.getcwd() + os.sep + out)
        #if platform == 'darwin':  # MacOS
            #os.system('open ' + save_path)

    #print('Done. (%.3fs)' % (time.time() - t0))