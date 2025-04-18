#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8 Training')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model yaml path or pretrained weights')
    parser.add_argument('--data', type=str, default='train.yaml', help='dataset yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='number of worker threads for data loading')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer: SGD, Adam, AdamW, etc.')
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (lr0 * lrf)')
    parser.add_argument('--patience', type=int, default=50, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--project', type=str, default='runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载模型
    model = YOLO(args.model)
    
    # 准备训练参数
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'patience': args.patience,
        'project': args.project,
        'name': args.name
    }
    
    # 开始训练
    model.train(**train_args)

if __name__ == '__main__':
    main() 