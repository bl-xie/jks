#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import hydra
from ultralytics.yolo.utils import DEFAULT_CONFIG
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.v8.detect.predict import init_tracker, predict
def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Detection with DeepSORT Tracking')
    parser.add_argument('--model', type=str, default='runs/train/exp5/weights/last.pt', help='model path or name')
    parser.add_argument('--source', type=str, default='test2.mp4', help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--save', action='store_true', help='save video tracking results')
    parser.add_argument('--project', type=str, default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', type=str, default='predict', help='save results to project/name')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 默认开启保存功能
    args.save = True
    
    # 初始化跟踪器
    init_tracker()
    
    # 转换参数为配置字典
    cfg_dict = vars(args)
    cfg_dict['imgsz'] = check_imgsz(args.imgsz, min_dim=2)
    
    # 使用hydra运行预测
    sys.argv = ['', f'model={args.model}', f'source={args.source}', 
                f'conf={args.conf}', f'iou={args.iou}', 
                f'imgsz={args.imgsz[0]}', f'device={args.device}', 
                f'show={args.show}', f'save={args.save}',
                f'project={args.project}', f'name={args.name}']
    
    # 调用predict()函数，不使用关键字参数
    predict()

if __name__ == '__main__':
    main() 