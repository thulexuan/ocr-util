"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import shutil

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

from craft import CRAFT

from collections import OrderedDict

    
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def parse_arguments():
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
    return args
    
    # args = parser.parse_args()

args = parse_arguments()

""" For test images in a folder """
# image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    # if cuda:
    #     x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

# crop original image theo bounding boxes and save cropped images in a new folder to ocr
def crop_and_save(original_image_path, bounding_boxes, output_folder):
    # Open the original image
    original_image = Image.open(original_image_path)

    # Iterate over bounding boxes and crop the image
    for i, box_points in enumerate(bounding_boxes):
        # Unpack coordinates of the four points
        x_min, y_min, x_max, y_max = box_points

        # Find the bounding box coordinates
        left = x_min 
        upper = y_min 
        right = x_max 
        lower = y_max 

        # Crop the image using the bounding box
        cropped_image = original_image.crop((left, upper, right, lower))

        # Save the cropped image to the output folder
        output_path = os.path.join(output_folder, f"cropped_{i+1}.png")
        cropped_image.save(output_path)
        # print(f"Cropped image saved at: {output_path}")


def text_detection_by_craft(image_path):

    # load net
    net = CRAFT()     # initialize

    # print('Loading weights from checkpoint (' + args.trained_model + ')')
    net.load_state_dict(copyStateDict(torch.load('craft_mlt_25k.pth', map_location=torch.device('cpu'))))
    
    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    # print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
    # print(image_path) # test/test_ocr.png
    image = imgproc.loadImage(image_path)

    bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    # cho y3 == y4
    height_of_words = []
    for bbox in bboxes:
        y_min = min(bbox[0][1], bbox[1][1])
        y_max = max(bbox[2][1], bbox[3][1])
        if bbox[2][1] != bbox[3][1]:
            bbox[2][1] = y_max
            bbox[3][1] = y_max
        # tính chiều cao min nhất trong các từ
        height_of_words.append(y_max - y_min)
         
    min_space_between_lines = min(height_of_words)
    print(min_space_between_lines)       
    bboxes = sorted(bboxes, key=lambda x: x[-1][1])
    #print(bboxes)

    lines_array = []

    # Initialize the first subarray with the first element
    current_subarray = [bboxes[0]]

    # Iterate through the sorted array
    for i in range(1, len(bboxes)):
        # Check if the difference is smaller than 10
        if bboxes[i][-1][1] - current_subarray[0][-1][1] < min_space_between_lines + 10:
            current_subarray.append(bboxes[i])
        else:
            # If the difference is larger, start a new subarray
            lines_array.append(current_subarray)
            current_subarray = [bboxes[i]]

        # Add the last subarray to the result
    lines_array.append(current_subarray)

        # print(lines_array[0])

    crop_points = []
        # mỗi line có một bộ (x_min, y_min, x_max, y_max)
    for line in lines_array:
        y_max_array = [point[2][1] for point in line]
        y_max_array += [point[3][1] for point in line]
        y_max = max(y_max_array)

        y_min_array = [point[0][1] for point in line]
        y_min_array += [point[1][1] for point in line]
        y_min = min(y_min_array)

        x_max_array = [point[1][0] for point in line]
        x_max_array += [point[2][0] for point in line]
        x_max = max(x_max_array)

        x_min_array = [point[0][0] for point in line]
        x_min_array += [point[3][0] for point in line]
        x_min = min(x_min_array)

        crop_points.append((x_min, y_min, x_max, y_max))
    
        # mỗi dòng có một điểm top-left và bottom-right để crop
    print(crop_points)

        # Create the output folder if it doesn't exist
    os.makedirs('./output_images', exist_ok=True)

    # delete existed images in folder
    files = os.listdir('./output_images')
    image_extensions=['.png', '.jpg', '.jpeg']

    # Filter files to keep only images
    image_files = [file for file in files if any(file.lower().endswith(ext) for ext in image_extensions)]

    # Delete each image file
    for image_file in image_files:
        image_path_in_folder = os.path.join('./output_images', image_file)
        os.remove(image_path_in_folder)

    # Call the function to crop and save
    crop_and_save(image_path, crop_points, './output_images')
    
    print("elapsed time : {}s".format(time.time() - t))
                
        
# ocr từng ảnh trong folder các ảnh sau khi crop
def process_images_in_folder(folder_path):

    s = ''

    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu' # device chạy 'cuda:0', 'cuda:1', 'cpu'

    detector = Predictor(config)

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter files to keep only images (you can customize the extensions)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # Process each image
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Perform your image processing or analysis here
        print(f"Processing image: {image_path}")

        img = Image.open(image_path)
        recognized_text_in_line = detector.predict(img)
        s += recognized_text_in_line
        s += '\n'
    return s

async def read_image(image_path):
    text_detection_by_craft(image_path)

    recognized_text = process_images_in_folder('./output_images')
    
    return recognized_text

def main():
    args = parse_arguments()
    text_detection_by_craft('test/test_ocr.png')
    recognized_text = process_images_in_folder('./output_images')
    print(recognized_text)

if __name__ == "__main__":
    main()



    



    
