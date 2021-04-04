import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from .data import DataIterator
from .dali import DaliDataIterator
from .model import Model
from .utils import Profiler
from .utils import show_detections
import mmcv
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
from pathlib import Path
import shutil
from time import* 
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_boxes(img, bbox, cls, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color_txt = compute_color_for_labels(id)
        color_rec = mmcv.visualization.color_val('green')
     #   import pdb;pdb.set_trace()
        label = '{}{:d} cls={}'.format("", id, cls[i])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (0, 150, 0), -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [256, 256, 256], 2)
    return img

def draw_boxes_anno(img, bbox, cls, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color_txt = compute_color_for_labels(id)
        color_rec = mmcv.visualization.color_val('red')
        label = '{}{:d}cls={}'.format("", id, cls[i])
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color_rec, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color_rec, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def infer(model, path, detections_file, resize, max_size, batch_size, mixed_precision=False, is_master=True, world=0, annotations=None, use_dali=True, is_validation=False, verbose=True):
    'Run inference on images from path'

    print('model',model)
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    #print("backend",backend)
    stride = model.module.stride if isinstance(model, DDP) else model.stride
    
    # Create annotations if none was provided
    file_pth = os.listdir(path)
    file_pth.sort()
    if not annotations:
        annotations = tempfile.mktemp('.json')
        images = [{ 'id': i, 'file_name': f} for i, f in enumerate(file_pth)]
        json.dump({ 'images': images }, open(annotations, 'w'))

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare dataset
    if verbose: print('Preparing dataset...')

    data_iterator = (DaliDataIterator if use_dali else DataIterator)(
        path, resize, max_size, batch_size, stride,
        world, annotations, training=False)
    if verbose: print(data_iterator)
    
    # Prepare model
    if backend is 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.cuda()
            model = amp.initialize(model, None,
                               opt_level = 'O2' if mixed_precision else 'O0',
                               keep_batchnorm_fp32 = True,
                               verbosity = 0)
   
   
        model.eval()

    

    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    out = 'inference/output'
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'gpu' if world == 1 else 'gpus'))
        print('     batch: {}, precision: {}'.format(batch_size,
            'unknown' if backend is 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print('Running inference...')

    results = []
    profiler = Profiler(['infer', 'fw'])
    
    ## load annotation file XXXX.txt
    anno_file = open('VisDrone2019-MOT-val/annotations/uav0000305_00000_v.txt')
    anno = anno_file.readlines()
    anno_id = []
    bbox_anno = []
    obj_anno = []
    for nk in range(len(data_iterator)):
        anno_id.append([])
        bbox_anno.append([])
        obj_anno.append([])
    for ab, anno_it in enumerate(anno):
        idd = int(anno_it.split(',')[0])
        bbox_anno[idd-1].append([float(anno_it.split(',')[2]), float(anno_it.split(',')[3]), float(anno_it.split(',')[4]), float(anno_it.split(',')[5])])
        anno_id[idd-1].append(float(anno_it.split(',')[7]))
        obj_anno[idd-1].append(float(anno_it.split(',')[1]))
  #  anno_id = np.array(anno_id)
  #  bbox_anno = np.array(bbox_anno)
    prop_num = np.array([0])
    with torch.no_grad():
        for i, (data, ids, ratios) in enumerate(data_iterator):
            # Forward pass
            profiler.start('fw')        
            scores, boxes, classes = model(data)
            profiler.stop('fw')
            img = np.array(data[0].permute([1, 2, 0]).cpu())
           
            results.append([scores, boxes, classes, ids, ratios])
            cls_pt = np.array(classes[0].cpu())
            prop_num += boxes.shape[1]
    
            # person in coco: 0-person
            cls_pt[np.where(cls_pt==0)]=1
            # vehicle in coco: 1-bike; 2-car; 3-motor; 5-bus; 7-trunk
            cls_pt[np.where((cls_pt==1)|(cls_pt==2)|(cls_pt==3)|(cls_pt==5)|(cls_pt==7))]=2
            cls_pt[np.where((cls_pt!=1)&(cls_pt!=2))]=0   

            anno_iid = np.array(anno_id[i])
            # person in VisDrone: 1-pedestrian; 2-people
            anno_iid[np.where((anno_iid==1)|(anno_iid==2))]=1
            # vehicle in VisDrone: 3-bike; 4-car; 5-van; 6-trunk; 7-tricycle; 8-awning-tricycle; 9-bus; 10-motor
            anno_iid[np.where((anno_iid==3)|(anno_iid==4)|(anno_iid==5)|(anno_iid==6)|(anno_iid==7)|(anno_iid==8)|(anno_iid==9)|(anno_iid==10))]=2
            anno_iid[np.where((anno_iid!=1)&(anno_iid!=2))]=0
  
            
            bbox_id = np.array(bbox_anno[i])   
            bbox_xy = np.zeros_like(bbox_id) + bbox_id

            bbox_xy[:, 2] = bbox_id[:, 0] + bbox_id[:, 2] 
            bbox_xy[:, 3] = bbox_id[:, 1] + bbox_id[:, 3]
            bbox_xy *= np.array(ratios.cpu()).item()

            obj_id = np.array(obj_anno[i]) + 1

            
            bbox_xywh = []     
            cbox = boxes[0].cpu()
            for kk, box in enumerate(cbox):
                x_c, y_c, bbox_w, bbox_h = bbox_rel(box)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)

            xywhs = torch.Tensor(bbox_xywh)
           
            track_begin = time()
            outputs = deepsort.update(xywhs, scores[0].cpu(), img)
            track_end = time()

            # change the image brightness
            row, col, cha = img.shape
            blank = np.zeros([row, col, cha], img.dtype)
            imgb = cv2.addWeighted(img, 2, blank, 1, 50)

            if len(outputs) > 0:
                 bbox_xyxy = outputs[:, :4]
                 identities = outputs[:, -1]
                 draw_boxes_anno(imgb, bbox_xy, anno_iid, obj_id)
                 draw_boxes(imgb, bbox_xyxy, cls_pt, identities)
            if len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                    #    import pdb;pdb.set_trace()
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (i, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, scores[0][j].cpu().item(), -1, -1, -1))
            
          #  cv2.imwrite('VisDrone_det_6/img{:0>7d}.jpg'.format(i+1), imgb)
            
            profiler.bump('infer')
            if verbose and (profiler.totals['infer'] > 60 or i == len(data_iterator) - 1):
                size = len(data_iterator.ids)
                msg  = '[{:{len}}/{}]'.format(min((i + 1) * batch_size,
                    size), size, len=len(str(size)))
                msg += ' {:.3f}s/{}-batch'.format(profiler.means['infer'], batch_size)
                msg += ' (dt: {:.3f}s)'.format(profiler.means['fw'])
                msg += ' (tk: {:.3f}s)'.format(track_end - track_begin)
                msg += ' (prop_num: {:.0f}/im)'.format(int(prop_num.item()/i))
                msg += ' (input_rsl: {:.0f}*{:.0f})'.format(img.shape[0], img.shape[1])
                msg += ' (threshold: {:.3f})'.format(0.05)
                msg += ', {:.1f} im/s'.format(batch_size / profiler.means['infer'])

                print(msg, flush=True)

                profiler.reset()
   

    # Gather results from all devices
    if verbose: print('Gathering results...')
    results = [torch.cat(r, dim=0) for r in zip(*results)]
    if world > 1:
        for r, result in enumerate(results):
            all_result = [torch.ones_like(result, device=result.device) for _ in range(world)]
            torch.distributed.all_gather(list(all_result), result)
            results[r] = torch.cat(all_result, dim=0)

    if is_master:
        # Copy buffers back to host
        results = [r.cpu() for r in results]

        # Collect detections
        detections = []
        processed_ids = set()
             
             
        for scores, boxes, classes, image_id, ratios in zip(*results):
            image_id = image_id.item()
            if image_id in processed_ids:
                continue
            processed_ids.add(image_id)
        
        
            keep = (scores > 0).nonzero()
            scores = scores[keep].view(-1)
            boxes = boxes[keep, :].view(-1, 4) / ratios
            classes = classes[keep].view(-1).int()
            #print('classes', classes)
            for score, box, cat in zip(scores, boxes, classes):
                x1, y1, x2, y2 = box.data.tolist()
                cat = cat.item()
                if 'annotations' in data_iterator.coco.dataset:
                    cat = data_iterator.coco.getCatIds()[cat]
                    #if cat !=3:
                      #continue
                    #print('cat',cat)
                detections.append({
                    'image_id': image_id,
                    'score': score.item(),
                    'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                    'category_id': cat
                })
                #show_detections(detections)
        
        if detections:
            # Save detections
            if detections_file and verbose: print('Writing {}...'.format(detections_file))
            detections = { 'annotations': detections }
            detections['images'] = data_iterator.coco.dataset['images']
            if 'categories' in data_iterator.coco.dataset:
                detections['categories'] = [data_iterator.coco.dataset['categories']]
            if detections_file:
                json.dump(detections, open(detections_file, 'w'), indent=4)

            # Evaluate model on dataset
            if 'annotations' in data_iterator.coco.dataset:
                if verbose: print('Evaluating model...')
                with redirect_stdout(None):
                    coco_pred = data_iterator.coco.loadRes(detections['annotations'])
                    coco_eval = COCOeval(data_iterator.coco, coco_pred, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                coco_eval.summarize()
        else:
            print('No detections!')
