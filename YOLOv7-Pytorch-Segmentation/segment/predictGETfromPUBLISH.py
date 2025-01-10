#!/usr/bin/env python
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python segment/predict.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python segment/predict.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg.xml                # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          
！每次都要改：pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)的maxdet
"""
from std_msgs.msg import String
import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import torch.backends.cudnn as cudnn
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import signal
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
import cv2

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(720, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold   conf_thres=0.25
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image  max_det=1000
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader    摄像头输入 /从文件或文件夹加载图像
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=9, nm=32)
            #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)原来
            
            
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
           
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            LIST=[]
            if len(det):
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC tensor

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print det values (for debugging)
                print("det!!!", [int(det[:, 5][i]) for i in range(min(len(det[:, 5]), 9))])

                # Detect duplicates and missing numbers
                numbers = [int(det[:, 5][i]) for i in range(len(det[:, 5]))]
                print("Original numbers:", numbers)

                unique_numbers = set(numbers)
                repeated_numbers = [num for num in unique_numbers if numbers.count(num) > 1]
                missing_numbers = [i for i in range(9) if i not in numbers]

                # Find all repeated positions for repeated numbers
                repeated_positions = []
                for num in repeated_numbers:
                    positions = [idx for idx, value in enumerate(numbers) if value == num]
                    repeated_positions.extend(positions)

                # Handle repeated positions and missing numbers
                if repeated_positions and missing_numbers:
                    # Sort repeated positions so we replace them in order
                    repeated_positions.sort()

                    # Loop through repeated positions and replace them with missing numbers
                    missing_idx = 0
                    for position in repeated_positions:
                        if missing_idx < len(missing_numbers):
                            det[:, 5][position] = torch.tensor(missing_numbers[missing_idx], dtype=det[:, 5].dtype)
                            missing_idx += 1
                        else:
                            break  # Stop if there are no more missing numbers

                    # Debugging final det
                    print("det after fixing:", [int(det[:, 5][i]) for i in range(len(det[:, 5]))])

                else:
                    print("No repeated positions or missing numbers found.")

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]  # RGB[(56, 56, 255)]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh, imw, 3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w



                #print("im[i]:",im[i])#!!!
                #print("masks:",masks.shape)#!!! mask是一个和元图像一样大的tensor，他觉得是的像素为1，不是的像素为2
                #cv2.imshow('image', im_masks)
                #cv2.waitKey(1)

                
                # Mask plotting ----------------------------------------------------------------------------------------
                """
                zi=1

                # Write results                         print("xyxy",xyxy)#!!!
                for *xyxy, conf, cls in reversed(det[:, :6]):#det[:, :6]==xyxy，可信度，引索 

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                       
                        
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    formatted_xyxy = format_xyxy(xyxy)
                    #print(label.split()[0],formatted_xyxy)#!!!((983, 3014), (2438, 3504))
                    #print("type",type(formatted_xyxy[0][0]))#<class 'int'>
                    
                    list=[[label.split()[0]],[formatted_xyxy[0][0],formatted_xyxy[0][1]],[formatted_xyxy[1][0],formatted_xyxy[1][1]]]
                    LIST.append(list)
                    """
                #-------------------------------------------------------------------------------------------
                
                
                
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]#RGB[(56, 56, 255)]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3) 画红色范围
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                #print("im[i]:",im[i])#!!!
                #print("masks:",masks.shape)#!!! mask是一个和元图像一样大的tensor，他觉得是的像素为1，不是的像素为2
                #cv2.imshow('image', im_masks)
                #cv2.waitKey(1)

                
                # Mask plotting ----------------------------------------------------------------------------------------
                
                zi=1

                # Write results                         print("xyxy",xyxy)#!!!
                for *xyxy, conf, cls in reversed(det[:, :6]):#det[:, :6]==xyxy，可信度，引索 

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                       
                        
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#有置信度
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} ')#无置信度
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    formatted_xyxy = format_xyxy(xyxy)
                    #print(label.split()[0],formatted_xyxy)#!!!((983, 3014), (2438, 3504))
                    #print("type",type(formatted_xyxy[0][0]))#<class 'int'>
                    
                    list=[[label.split()[0]],[formatted_xyxy[0][0],formatted_xyxy[0][1]],[formatted_xyxy[1][0],formatted_xyxy[1][1]]]
                    LIST.append(list)
                    
                    ##pub = rospy.Publisher('coordinates', Float64MultiArray, queue_size=10)
    # 创建两个 Point 消息对象分别表示两个坐标点
                    ##coordinates_msg = Float64MultiArray()
    # 发布消息
                   ##coordinates_msg.data.extend([zi])
                    ##for point in formatted_xyxy:
                        ##coordinates_msg.data.extend(point)
                    ##zi+=1
                    ##rospy.loginfo("Publishing coordinates: %s", coordinates_msg.data)
                    ##pub.publish(coordinates_msg)
            
            pub = rospy.Publisher('YOLOv7_topic', String, queue_size=10)
            pub.publish(str(LIST))
            print("LIST",LIST)  
            # Stream results
            im0 = annotator.result()
            #print("!!!!!!!!!!!!!!!!",im0.shape[1], im0.shape[0])
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    #print("!!!!!!!!!!!!!!!!",im0.shape[1], im0.shape[0])
               # cv2.imshow(str(p), im0)
               # cv2.waitKey(1)  # 1 millisecond#窗口显示

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    #LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[720,1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
#--------------------------------
def format_xyxy(xyxy):
    """
    将张量列表转换为((x1, y1), (x2, y2))格式的元组
    Args:
        xyxy (list): 张量列表，形状为[tensor(x1), tensor(y1), tensor(x2), tensor(y2)]

    Returns:
        tuple: 格式化后的元组
    """
    xyxy_tuples = [(int(xyxy[i].item()), int(xyxy[i+1].item())) for i in range(0, len(xyxy), 2)]
    return tuple(xyxy_tuples)
#--------------------------------
def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    i = 0
    signal.signal(signal.SIGINT, my_sigint_handler)
    rospy.init_node('coordinate_publisher_once', anonymous=True)
    while i < 50000:
        print(i)
        i += 1
        try:
            run(**vars(opt))
            time.sleep(1)
        except Exception as e:
            
            continue  # 继续下一个图像
            
        
    # 用户交互
      #  user_input = input("输入 1 继续，输入 q 退出: ").strip().lower()
       # if user_input == 'q':
        #    print("退出程序...")
        #    break
       # elif user_input == '1':
       #     print("继续处理...")
       #     continue
       # else:
       #     print("无效输入，默认继续处理...")
        
        
        

def my_sigint_handler(signal, frame):
    print("Caught SIGINT signal, exiting gracefully.")
    # 在这里执行您希望在收到 SIGINT 信号时执行的操作
    # 比如关闭文件、释放资源等
    sys.exit(0)  # 退出程序
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

