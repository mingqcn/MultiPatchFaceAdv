"""
它将显示标签位置(初始化参数)默认参数。更改参数，直到图像看起来像带贴纸的已准备好的图像
"""

import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../Demo/'))
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import shutil
import skimage.io as io
from skimage import transform as trans

from prepare import detect_face

# Align face as ArcFace template--相似变换矩阵操作（平移、旋转、均尺度缩放）
def preprocess(img, landmark):
    image_size = [600,600]
    src = 600./112.*np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32)
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]

    warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
    return warped

def main(args):
    tf.disable_eager_execution()

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    sess = tf.Session()
    pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    threshold = [ 0.6, 0.7, 0.7 ]
    factor = 0.709

    dirs = os.listdir(args.input_path)
    for folder in dirs:
        path = os.path.join(args.input_path, folder)
        if not os.path.isdir(path):
            continue

        img_names = tf.gfile.Glob(os.path.join(path, '*.jpg'))
        if len(img_names) < 7:
            continue

        print('========= %s ===========' %(folder))
        output_path = os.path.join(args.output_path, folder)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        max_num = 8
        has001 = False
        for name in img_names:
            if max_num ==0 :
                break
            max_num = max_num - 1

            if max_num == 0 and not has001:
                names =  tf.gfile.Glob(os.path.join(path, '*_0001.jpg'))
                name = names[0]

            print('%s'%(os.path.basename(name)))
            img = io.imread(name)
            _minsize = min(min(img.shape[0]//5, img.shape[1]//5),80)
            bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
            assert bounding_boxes.size>0
            points = points[:, 0]
            landmark = points.reshape((2,5)).T
            warped = preprocess(img, landmark)
            output_name = name.replace(args.input_path,args.output_path)
            io.imsave(output_name,warped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='../data/lfw',
                        type=str, help='Path to the image.' )  # p6;先用[250，250]的维度来进行操作
    parser.add_argument('--output_path', default='../data/aligned',
                        type=str, help='Path to the image.')

    args = parser.parse_args()

    main(args)

