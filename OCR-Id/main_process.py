# -*- coding: utf-8 -*-
"""
输出结果为：
    CCFTestResultFixValidData_release.csv
"""

import argparse
import os
import sys
import time

import cv2

sys.path.append('./')

from recognize_process.tools import test_crnn_jmz
from data_correction_and_generate_csv_file.generate_test_csv_file import generate_csv
from find_split_img.find_img import main as find_img
from find_split_img.find_and_split_img import find_split_process
from find_split_img.cfg import fan, zheng, srcTri, t_imgs, template_match
from find_split_img.extract_test_img_to_txts import generate_txts
from find_split_img.preprocess_for_test import preprocess_imgs

def recoginze_init_args():
    """
    初始化识别过程需要的参数
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-rc_w', '--recognize_weights_path', type=str,
                        help='Path to the pre-trained weights to use',
                        default='./recognize_process/model_save/recognize_model')
    parser.add_argument('-rc_c', '--recognize_char_dict_path', type=str,
                        help='Directory where character dictionaries for the dataset were stored',
                        default='./recognize_process/char_map/char_map.json')
    parser.add_argument('-rc_i', '--recognize_image_path', type=str,
                        help='Path to the image to be tested',
                        default='./recognize_process/test_imgs/')
    parser.add_argument('-rc_t', '--recognize_txt_path', type=str,
                        help='Whether to display images',
                        default='./recognize_process/image_list.txt')
    parser.add_argument("--no_gen_data_chu", action="store_true", help="generate chusai new test data")
    parser.add_argument("--no_gen_data_fu", action="store_true", help="generate fusai new test data")
    parser.add_argument("--no_preprocessed", action="store_true", help="if preprocessed test data")
    parser.add_argument("--no_gan_test", action="store_true", help="test data with gan model")
    parser.add_argument("--no_gan_test_rematch", action="store_true", help="test rematch data with gan model")
    parser.add_argument("--no_rec_img", action="store_true", help="if recover img")
    parser.add_argument("--no_rec_img_rematch", action="store_true", help="if recover img")
    parser.add_argument("--no_test_data", action="store_true", help="if generate test data")
    parser.add_argument("--no_fix_img", action="store_true", help="if fix img of address and unit")
    parser.add_argument("--no_gen_txts", action="store_true", help="if txt files for recognize")
    parser.add_argument("--debug", action="store_true", help="if debug")
    parser.add_argument("--gan_chu", default="chusai_watermask_remover_model", help="model name of chusai")
    parser.add_argument("--gan_fu", default="fusai_watermask_remover_model", help="model name of fusai")
    parser.add_argument("--pool_num", default=-1, help="the number of threads for process data")
    parser.add_argument("--test_data_dir", required=True, help="the dir of test data")
    parser.add_argument("--test_experiment_name", required=True, help="the dir of test data")
    parser.add_argument("--gan_ids", required=True, help="-1 for cpu, 0 or 0,1.. for GPU")

    return parser.parse_args()


if __name__ == '__main__':

    args = recoginze_init_args()
    origin_img_path = args.test_data_dir
    time_log = time.strftime("%y_%m_%d_%H_%M_%S")
    header_dir = os.path.join("./data_temp", args.test_experiment_name + "_" + time_log)
    if not os.path.exists(header_dir):
        os.makedirs(header_dir)
    cut_twisted_save_path = os.path.join(header_dir, 'data_cut_twist')  # 切分、旋转后数据保存路径

    recognize_txt_path = os.path.join(header_dir, "test_data_txts")
    recognize_image_path = os.path.join(header_dir, "test_data_preprocessed")
    split_img_path = os.path.join(header_dir, "img_split")
    # (img_path, save_path, t_imgs, srcTri, template_match, fan, zheng)

    t1 = time.time()
    # 图片分隔 先分格成两个图片，身份证的前后面
    find_split_process(origin_img_path, split_img_path, t_imgs, srcTri, template_match)
    # 把分隔好的图片，处理成高度一致的图片
    preprocess_imgs(split_img_path, recognize_image_path, 0)

    # 把分隔好的图片写成txt文件
    generate_txts(origin_img_path, recognize_image_path, recognize_txt_path, 0)
    # ～～～～～～～～～～～～～～
    # recognize_image_path = r"F:\python_git_file\MY_OCR_ID_test_function\image\saveimg"
    # generate_txts(origin_img_path, recognize_image_path, recognize_txt_path, 0)
    # ～～～～～～～～～～～～～～
    # 识别图片
    test_crnn_jmz.recognize_jmz(image_path=recognize_image_path, weights_path=args.recognize_weights_path,
                                char_dict_path=args.recognize_char_dict_path, txt_file_path=recognize_txt_path)
    # print(time.time() - t1)
    origin_watermask_removed_img_path = os.path.join(header_dir, "recover_image_fu_dir")
    # recognize_txt_path = r"F:\python_git_file\ID\OCR-Id\data_temp\test_example_20_03_17_13_52_13\test_data_txts"
    # 把识别的结果进行矫正和存放在同一个csv文件里面
    generate_csv(origin_watermask_removed_img_path, recognize_txt_path, "./")
