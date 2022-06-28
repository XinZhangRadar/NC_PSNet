from __future__ import absolute_import

import numpy as np
import torch
import pdb;
def nms_cpu(dets, thresh):
    print('start Modified NMS')
    #pdb.set_trace();
    dets = dets.cpu().numpy();
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    
    p_box = [];#poitive box
    n_box = [];#negtive box
    i = 0;

    for score in scores:
        if score < 0.5:
            n_box.append(i);
            i = i + 1;
        else:
            p_box.append(i);
            i = i + 1;

    n_box = np.array(n_box);
    reward = np.zeros(len(p_box));
    penalize = np.zeros(len(p_box));


    if len(p_box) >1 :
        for i in p_box:
            beside_i = np.array(p_box[0:p_box.index(i)] + p_box[p_box.index(i)+1:len(p_box)]);
            xx1 = np.maximum( x1[i], x1[beside_i]);
            yy1 = np.maximum(y1[i], y1[beside_i]);
            xx2 = np.minimum(x2[i], x2[beside_i]);
            yy2 = np.minimum(y2[i], y2[beside_i]);
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[beside_i] - inter)
            inds = np.where(ovr > thresh)[0]
            reward[i] = np.sum(scores[beside_i[inds]]*ovr[inds]);
            if n_box.size != 0:
                xxx1 = np.maximum( x1[i], x1[n_box]);
                yyy1 = np.maximum(y1[i], y1[n_box]);
                xxx2 = np.minimum(x2[i], x2[n_box]);
                yyy2 = np.minimum(y2[i], y2[n_box]);
                ww = np.maximum(0.0, xxx2 - xxx1 + 1);
                hh = np.maximum(0.0, yyy2 - yyy1 + 1);
                inter2 = ww * hh;
                ovr2 = inter2 / (areas[i] + areas[n_box] - inter2)
                inds2 = np.where(ovr2 > (1-thresh))[0];
                penalize[i] =  np.sum((1-scores[n_box[inds2]])*ovr2[inds2]);
        scores_p = reward - penalize;
        dets_p = dets[p_box];
        dets_p[:, 4] = scores_p;
        keep_r = regular_nms(dets_p, thresh);
        p_box = np.array(p_box);
        keep = p_box[keep_r];

    else:
        keep = regular_nms(dets, thresh);
    return torch.IntTensor(keep)


def regular_nms(dets, thresh):
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

