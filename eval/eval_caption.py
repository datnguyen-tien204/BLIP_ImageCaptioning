import os

# from pycocotools.coco import COCO
# from pycocoevalcap.eval import COCOEvalCap
from .coco_caption.pycocotools.coco import COCO
from .coco_caption.pycocoevalcap.eval import COCOEvalCap

import torch
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval

def uit_viic_caption_eval(gt, results_file, split):
    # create cap object and cap_result object
    files = {
        'val': 'uitviic_captions_val2017.json',
        'test': 'uitviic_captions_test2017.json'
    }
    gt_file = os.path.join(gt, files[split])
    cap = COCO(gt_file)
    cap_result = cap.loadRes(results_file)
    # create cap_eval object by taking cap and cap_result
    cap_eval = COCOEvalCap(cap, cap_result)

    # evaluate on a subset of images by setting
    # cap_eval.params['image_id'] = cap_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # cap_eval.params['image_id'] = cap_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cap_eval.evaluate()

    # print output evaluation scores
    for metric, score in cap_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return cap_eval