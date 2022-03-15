import os
import torch
import argparse
import numpy as np
from PIL import Image

import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder, augmented_model_path='augmented_model.onnx'):
        self.image_folder = calibration_image_folder
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        cuda = torch.cuda.is_available()
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, providers=self.providers)
            (_, _, height, width) = session.get_inputs()[0].shape
            nchw_data_list = preprocess_func(self.image_folder, height, width, size_limit=0)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nchw_data_list)
            self.enum_data_dicts = iter([{input_name: nchw_data} for nchw_data in nchw_data_list])
        return next(self.enum_data_dicts, None)

def preprocess_func(images_folder, height, width, size_limit=0):
    '''
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    '''
    image_names = os.listdir(images_folder)
    image_names = [ i for i in image_names if i.endswith("jpg")]
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        # pillow_img = Image.new("RGB", (width, height))
        img = cv2.imread(image_filepath)
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        # input_data = np.float32(pillow_img) - \
        # np.array([123.68, 116.78, 103.94], dtype=np.float32)
        # input_data = np.array(pillow_img).astype('int64')
        nhwc_data = np.expand_dims(input_data, axis=0)
        nchw_data = np.transpose(nhwc_data, (0, 3, 1, 2))
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    return batch_data

def main():
    parser = argparse.ArgumentParser(description='Do static quantization for YOLOv5 onnx models.')
    parser.add_argument('-i', '--input-model', type=str, default="yolov5s.onnx", help='Input ONNX model path.')
    parser.add_argument('-o', '--output-model', type=str, default="yolov5s-static-quantized.onnx", help='Output model path.')
    parser.add_argument('-c', '--calibration-data', type=str, default="datasets/coco/images/val2017/", help='Images folder for data calibration.')
    args = parser.parse_args()

    dr = DataReader(args.calibration_data)
    quantize_static(args.input_model, args.output_model, dr)

    print('Calibrated and quantized model saved to: {}'.format(args.output_model))

if __name__ == '__main__':
    main()