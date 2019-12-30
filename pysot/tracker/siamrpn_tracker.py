# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker


class SiamRPNTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()

        self.center_pos = []
        self.size = []
        self.channel_average = []

        ##########다중처리를 위한 부분##########
        self.tracks = []  # tracking object lists
        self.id = 0  # tracking ids

    # template 추가를 위한 함수
    def addlist(self, _class, _img, _bbox):

        _id = self.id
        self.id = self.id + 1

        # _class = _class.cpu().numpy()

        # for i, box in enumerate(_bbox):
        #     _bbox[i] = box.cpu().numpy()

        _center_pos = np.array([_bbox[0] + (_bbox[2] - 1) / 2, _bbox[1] + (_bbox[3] - 1) / 2])
        _size = np.array([_bbox[2], _bbox[3]])

        # calculate z crop size
        w_z = _size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(_size)  # CONTEXT_AMOUNT : 0.5
        h_z = _size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(_size)
        s_z = round(np.sqrt(w_z * h_z))  # 넓이

        # calculate channel average
        _channel_average = np.mean(_img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(_img, _center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, _channel_average)
        zf = self.model.template(z_crop)

        self.tracks.append({"id": _id, "class": _class, "bbox": _bbox, "center_pos": _center_pos, "size": _size,
                            "channel_average": _channel_average, "template": zf, "threshold_cnt": 0, "best_score": 0})

    # template 삭제를 위한 함수
    def deletelist(self, _id):
        for i, _track in enumerate(self.tracks):
            if _track["id"] == _id:
                del self.tracks[i]

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.bSize = len(bbox)  # 박스의 개수 파악
        # print("size :", self.bSize)

        for i in range(self.bSize):
            self.center_pos.append(np.array([bbox[i][0] + (bbox[i][2] - 1) / 2,
                                             bbox[i][1] + (bbox[i][3] - 1) / 2]))
            self.size.append(np.array([bbox[i][2], bbox[i][3]]))

            # calculate z crop size
            w_z = self.size[i][0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size[i])  # CONTEXT_AMOUNT : 0.5
            h_z = self.size[i][1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size[i])
            s_z = round(np.sqrt(w_z * h_z))  # 넓이

            # calculate channle average
            self.channel_average.append(np.mean(img, axis=(0, 1)))

            # get crop
            z_crop = self.get_subwindow(img, self.center_pos[i],
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average[i])
            self.model.template(z_crop)

    # 이부분은 다중처리를 위한 작업 완료
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        bboxs = []
        best_score = []

        for i in range(self.bSize):
            w_z = self.size[i][0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size[i])
            h_z = self.size[i][1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size[i])
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z  # exampler size : 127
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)  # instance_size : 255
            x_crop = self.get_subwindow(img, self.center_pos[i],
                                        cfg.TRACK.INSTANCE_SIZE,
                                        round(s_x), self.channel_average[0])

            outputs = self.model.track(x_crop, i)

            score = self._convert_score(outputs['cls'])
            pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

            def change(r):
                return np.maximum(r, 1. / r)

            def sz(w, h):
                pad = (w + h) * 0.5
                return np.sqrt((w + pad) * (h + pad))

            # scale penalty
            s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                         (sz(self.size[i][0] * scale_z, self.size[i][1] * scale_z)))

            # aspect ratio penalty
            r_c = change((self.size[i][0] / self.size[i][1]) /
                         (pred_bbox[2, :] / pred_bbox[3, :]))
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            pscore = penalty * score

            # window penalty
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx = np.argmax(pscore)

            bbox = pred_bbox[:, best_idx] / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

            cx = bbox[0] + self.center_pos[i][0]
            cy = bbox[1] + self.center_pos[i][1]

            # smooth bbox
            width = self.size[i][0] * (1 - lr) + bbox[2] * lr
            height = self.size[i][1] * (1 - lr) + bbox[3] * lr

            # clip boundary
            cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                    height, img.shape[:2])

            # udpate state
            self.center_pos[i] = np.array([cx, cy])
            self.size[i] = np.array([width, height])

            bboxs.append([cx - width / 2,
                          cy - height / 2,
                          width,
                          height])
            best_score.append(score[best_idx])

        return {
            'bbox': bboxs,
            'best_score': best_score
        }

    def darkTrack(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """

        for _track in self.tracks:
            w_z = _track["size"][0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(_track["size"])
            h_z = _track["size"][1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(_track["size"])
            s_z = np.sqrt(w_z * h_z)
            scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z  # exampler size : 127
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)  # instance_size : 255
            x_crop = self.get_subwindow(img, _track["center_pos"],
                                        cfg.TRACK.INSTANCE_SIZE,
                                        round(s_x), _track["channel_average"])

            outputs = self.model.darkTrack(x_crop, _track["template"])

            score = self._convert_score(outputs['cls'])
            pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

            def change(r):
                return np.maximum(r, 1. / r)

            def sz(w, h):
                pad = (w + h) * 0.5
                return np.sqrt((w + pad) * (h + pad))

            # scale penalty
            s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                         (sz(_track["size"][0] * scale_z, _track["size"][1] * scale_z)))

            # aspect ratio penalty
            r_c = change((_track["size"][0] / _track["size"][1]) /
                         (pred_bbox[2, :] / pred_bbox[3, :]))
            penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
            pscore = penalty * score

            # window penalty
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx = np.argmax(pscore)

            bbox = pred_bbox[:, best_idx] / scale_z
            lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

            cx = bbox[0] + _track["center_pos"][0]
            cy = bbox[1] + _track["center_pos"][1]

            # smooth bbox
            width = _track["size"][0] * (1 - lr) + bbox[2] * lr
            height = _track["size"][1] * (1 - lr) + bbox[3] * lr

            # clip boundary
            cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                    height, img.shape[:2])

            # udpate state
            _track["center_pos"] = np.array([cx, cy])
            _track["size"] = np.array([width, height])
            _track["bbox"] = [int(cx - width / 2), int(cy - height / 2), int(width), int(height)]
            _track["best_score"] = score[best_idx]

        return self.tracks
