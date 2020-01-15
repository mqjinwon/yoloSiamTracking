from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

##########################others import########################
import time
import pickle as pkl
from darknet import Darknet
from util import *

torch.set_num_threads(1)


'''darknet 관련 인자'''
parser = argparse.ArgumentParser(description='multi object detection using Siamese network')
parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.9)
parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
parser.add_argument("--reso", dest='reso',
                    help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                    default="160", type=str)

'''siamese 관련 인자'''
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument("--tracking_num", dest='tracking_num', type=int,
                    help="how many bounding box do you want to track", default=5)
parser.add_argument("--l2_threshold", dest='l2_threshold', type=float,
                    help="compare YOLOv3_bbox with siamese_bbox by L2 norm", default=200)
parser.add_argument("--coor_update", dest='coor_update', type=float,
                    help="update weight between YOLOv3_bbox and siamese_bbox", default=0.7)
parser.add_argument("--score_threshold", dest='score_threshold', type=float,
                    help="siamese score threshold", default=0.97)
parser.add_argument("--count_threshold", dest='count_threshold', type=int,
                    help="undetected number threshold", default=5)
parser.add_argument("--record", dest='record', type=bool,
                    help="video record", default=False)

args = parser.parse_args()

colors = pkl.load(open("pallete", "rb"))  # color num : 100, bbox color according to classes
classes = load_classes('data/coco.names')  # load file converting index to class name


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    dim = img.shape[1], img.shape[0]
    img = cv2.resize(img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)

    return img_, dim


# write letter
def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    confi = x[-3].cpu().numpy()
    confi = format(confi, "2.2f")

    label = "{0}".format(classes[cls])
    color = colors[cls]
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4

    if c1[0] == 0 and c1[1] == 0:
        pass
    else:
        cv2.rectangle(img, c1, (c2[0] + 30, c2[1]), color, -1)
        cv2.putText(img, str(label) + str(confi), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    [225, 255, 255], 1)

    return img

def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup -- 처음에 카메라 조리개?가 켜지는 시간을 기다리는 변수(초기값은 5였음)
        for i in range(10):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    confidence = float(args.confidence)  # 물체를 탐지할때의 임계값
    nms_thesh = float(args.nms_thresh)  # 물체를 중복인식 하지 않도록 하는 nms 임계값
    VIDEO = args.record  # 영상을 저장할지 말지 정하는 변수

    # STEP 2의 hyper parameter 결정하기
    L2_threshold = args.l2_threshold  # 박스들 사이의 거리를 비교함으로써 같은 박스인지 검출
    coor_update = args.coor_update  # 좌표값 갱신시 darknet bbox와 siam bbox의 가중치를 결정
    score_threshold = args.score_threshold  # siam의 유사도를 측정하는 threshold값
    count_threshold = args.count_threshold  # 미검출을 얼마나 했는지 측정하는 threshold값
    tmplate_num = int(args.tracking_num)  # 몇 개의 물체를 추적할지 정하는 파라미터

    # 영상 저장시 필요한 변수들
    if VIDEO:
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        darkout = cv2.VideoWriter("C:/Users/vision/Desktop/pysot/dark1.avi", fourcc, 20.0, (1280, 960))
        siamout = cv2.VideoWriter("C:/Users/vision/Desktop/pysot/siam1.avi", fourcc, 20.0, (1280, 960))

    CUDA = torch.cuda.is_available()

    num_classes = 80

    darknet = Darknet(cfgfile)
    darknet.load_weights(weightsfile)

    darknet.net_info["height"] = 416  # 32배수의 이미지이어야한다(YOLO를 위해) # 416 #192 #608 #args.reso
    inp_dim = int(darknet.net_info["height"])

    # 이미지 사이즈 제한 걸기
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        darknet.cuda()

    darknet.eval()

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu())["state_dict"])
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):

        start = time.time()

        '''STEP 1 : YOLOv3와 siamese network를 거침'''

        '''STEP 1-1 : YOLOv3 network 돌리기'''
        
        img,  dim = prep_image(frame, inp_dim)  # inp_dim으로 사이즈를 통일시킨다

        if CUDA:
            img = img.cuda()

        darkoutputs = darknet(img, CUDA)

        darkoutputs = write_results(darkoutputs, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        darkoutputs[:, 1:5] = torch.clamp(darkoutputs[:, 1:5], 0.0, float(inp_dim)) / inp_dim

        darkoutputs[:, [1, 3]] *= frame.shape[1]
        darkoutputs[:, [2, 4]] *= frame.shape[0]

        darkimg = frame.copy()

        list(map(lambda x: write(x, darkimg), darkoutputs))  # box 그려주고, class 써주는 것(output이 X로 들어가진다)

        '''STEP 1-2 : siamese network 돌리기'''
        
        siamimg = frame.copy()

        siamoutputs = tracker.darkTrack(siamimg)  # 'bbox': bbox, 'best_score': best_score

        '''STEP 2 : 알고리즘으로 bbox 갱신하기'''
        
        samebboxlist = []  # 같은 bbox의 id를 적는 리스트(나중에 threshold 비교용)

        # 객체 검출 bbox와 객체 추적 bbox 비교 과정
        for darkoutput in darkoutputs:
            darkclass = darkoutput[-1].cpu().numpy()
            darkbbox = [darkoutput[1].cpu().numpy(), darkoutput[2].cpu().numpy(),
                        (darkoutput[3] - darkoutput[1]).cpu().numpy(),
                        (darkoutput[4] - darkoutput[2]).cpu().numpy()]

            findFlag = False  # 같은 bbox를 찾았는지 확인용

            for siamoutput in siamoutputs:
                siamclass = siamoutput['class']
                siambbox = siamoutput['bbox']

                #L2 norm으로 비교를 한다.
                l2norm = np.sqrt((siambbox[0]-darkbbox[0])**2 + (siambbox[1]-darkbbox[1])**2 +
                                 (siambbox[2]-darkbbox[2])**2 + (siambbox[3]-darkbbox[3])**2)

                # print("l2 : ", l2norm)

                if (l2norm < L2_threshold) and (darkclass == siamclass):  # L2 norm안에 들어오고 같은 클래스라면
                    # 좌표 갱신시켜주기
                    siamoutput['bbox'][0] = coor_update * siambbox[0] + (1 - coor_update) * darkbbox[0]
                    siamoutput['bbox'][1] = coor_update * siambbox[1] + (1 - coor_update) * darkbbox[1]
                    siamoutput['bbox'][2] = coor_update * siambbox[2] + (1 - coor_update) * darkbbox[2]
                    siamoutput['bbox'][3] = coor_update * siambbox[3] + (1 - coor_update) * darkbbox[3]
                    siamoutput['center_pos'] = np.array([siamoutput['bbox'][0] + (siamoutput['bbox'][2] - 1) / 2,
                                                  siamoutput['bbox'][1] + (siamoutput['bbox'][3] - 1) / 2])

                    # threshold_cnt 초기화
                    siamoutput['threshold_cnt'] = 0

                    print("L2Norm : {:5.2f}".format(l2norm))

                    samebboxlist.append(siamoutput['id'])

                    findFlag = True
                    break  # 찾았기 때문에 그냥 탈출한다.

            # 새로운 템플릿 추가 과정
            if not findFlag:  # 겹치지 않는 bbox는 새로운 템플릿으로 추가시켜준다.
                if len(tracker.tracks) < tmplate_num:  # darknet 오류 예외처리(원인을 모르겠음...)
                    if darkbbox[0] == 0 and darkbbox[0] == 0:
                        pass
                    else:
                        print(darkbbox)
                        tracker.addlist(darkclass, frame, [darkbbox[0], darkbbox[1], darkbbox[2], darkbbox[3]])

        # 템플릿을 삭제하는 과정
        for siamoutput in siamoutputs:
            if siamoutput['id'] in samebboxlist:  # L2 norm안에 들어오지 않는다면
                pass
            else:
                if siamoutput['best_score'] > score_threshold:  # score_threshold 값 이상이라면
                    siamoutput['threshold_cnt'] = 0  # threshold_cnt 초기화

                else:
                    siamoutput['threshold_cnt'] += 1  # threshold_cnt 증가시키기

                    if siamoutput['threshold_cnt'] > count_threshold:
                        print("lost \"{0}\" object".format(classes[int(siamoutput['class'])]))
                        tracker.deletelist(siamoutput['id'])

        # 이미지 띄우기
        for siamoutput in siamoutputs:
            bbox = list(map(lambda x: int(x), siamoutput['bbox']))
            score = str(round(siamoutput['best_score'], 3))
            cls = int(siamoutput['class'])
            label = "{0}".format(classes[cls])
            color = colors[cls]

            cv2.putText(siamimg, label + score, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
            cv2.rectangle(siamimg, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          color, 3)

        cv2.imshow("darkimg", darkimg)
        cv2.imshow(video_name, siamimg)
        if VIDEO:
            darkout.write(darkimg)
            siamout.write(siamimg)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        print("FPS of the video is {:5.2f}".format(1 / (time.time() - start)))

    if VIDEO:
        darkout.release()
        siamout.release()

        # # #첫 프레임!!! GT 초기화 작업을 여기서 진행한다.
        # first_frame = True
        #
        # if first_frame:
        #     try:
        #         init_rect = cv2.selectROIs(video_name, frame, False, False) #초기 GT 설정하는 부분(selectROI : 한개, selectROIS : 여러개)
        #     except:
        #         exit()
        #     tracker.init(frame, init_rect) #build_tracker를 사용
        #     first_frame = False
        # else:
        #     start = time.time()
        #
        #     # ======================================================================================================
        #
        #     # darknet 돌리는 부분!!
        #     img, orig_im, dim = prep_image(frame, inp_dim)  # inp_dim으로 사이즈를 통일시킨다
        #
        #     if CUDA:
        #         img = img.cuda()
        #
        #     output = darknet(Variable(img), CUDA)
        #
        #     # print("feature shape : ", np.shape(feature))
        #
        #     output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        #
        #     output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
        #
        #     output[:, [1, 3]] *= frame.shape[1]
        #     output[:, [2, 4]] *= frame.shape[0]
        #
        #     list(map(lambda x: write(x, orig_im), output))  # box 그려주고, class 써주는 것(output이 X로 들어가진다)
        #
        #     cv2.imshow("frame", orig_im)
        #
        #     # ======================================================================================================
        #
        #     '''물체를 탐지한 후에 추적하게 만드는 소스'''
        #     # list(map(lambda x: tracker.addlist(x[-1], frame, [x[1], x[2], (x[3]-x[1])/2, (x[4]-x[2])/2]), output))
        #     #
        #     # outputs = tracker.darkTrack(frame)  # 'bbox': bbox, 'best_score': best_score
        #     #
        #     # for i in range(len(tracker.tracks)):
        #     #     if 'polygon' in outputs:  # 이건 siammask에서 사용...!!
        #     #         polygon = np.array(outputs['polygon']).astype(np.int32)
        #     #         cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
        #     #                       True, (0, 255, 0), 3)
        #     #         mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
        #     #         mask = mask.astype(np.uint8)
        #     #         mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
        #     #         frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
        #     #
        #     #     else:  # 이걸로 그리게 된다.(다중으로 바꿔줘야함.)
        #     #         bbox = list(map(int, outputs['bbox'][i]))
        #     #         score = str(round(outputs['best_score'][i], 3))
        #     #         cv2.putText(frame, score, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        #     #         cv2.rectangle(frame, (bbox[0], bbox[1]),
        #     #                       (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        #     #                       (0, 255, 0), 3)
        #     #
        #     # cv2.imshow(video_name, frame)
        #     # key = cv2.waitKey(1)
        #     # if key & 0xFF == ord('q'):
        #     #     break
        #     #
        #     # print("FPS of the video is {:5.2f}".format(1 / (time.time() - start)))
        #
        #     '''물체를 직접 지정해서 트래킹 하는 모드 관련 소스'''
        #     # outputs = tracker.track(frame)  # 'bbox': bbox, 'best_score': best_score
        #     #
        #     # for i in range(len(outputs['bbox'])):
        #     #     if 'polygon' in outputs:  # 이건 siammask에서 사용...!!
        #     #         polygon = np.array(outputs['polygon']).astype(np.int32)
        #     #         cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
        #     #                       True, (0, 255, 0), 3)
        #     #         mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
        #     #         mask = mask.astype(np.uint8)
        #     #         mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
        #     #         frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
        #     #
        #     #     else:  # 이걸로 그리게 된다.(다중으로 바꿔줘야함.)
        #     #         bbox = list(map(int, outputs['bbox'][i]))
        #     #         score = str(round(outputs['best_score'][i], 3))
        #     #         cv2.putText(frame, score, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
        #     #         cv2.rectangle(frame, (bbox[0], bbox[1]),
        #     #                       (bbox[0] + bbox[2], bbox[1] + bbox[3]),
        #     #                       (0, 255, 0), 3)
        #     #
        #     #
        #     # cv2.imshow(video_name, frame)
        #     # key = cv2.waitKey(1)
        #     # if key & 0xFF == ord('q'):
        #     #     break
        #     #
        #     # print("FPS of the video is {:5.2f}".format(1 / (time.time() - start)))


if __name__ == '__main__':
    main()
