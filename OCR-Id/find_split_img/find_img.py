# -*- coding: utf-8 -*-

# @Date    : 2020- 03- 05
# @Time    : 15: 10
# @Author  : Linear2pi

import os

import cv2
import numpy as np
from find_split_img.cfg import template_match, t_imgs, srcTri


def crop_img(ori_img, tempalte_size, save_path, img, label):
    """
    :param ori_img: 图片
    :param tempalte_size: 元素相关参数,坐标,长宽
    :param save_path: 切割之后的元素保存路径
    :param seq: 序号
    :param label: 标记
    :param type_c: 类型(没有用到)
    :return:
    """
    try:
        x_p = tempalte_size["x_d"]
        y_p = tempalte_size["y_d"]
        c_img = ori_img[y_p:y_p + tempalte_size["h"], x_p: x_p + tempalte_size["w"]]
        c_img_save_path = os.path.join(save_path, "%s_%s_%s.jpg" % (img, label, str(tempalte_size["index"])))
        cv2.imwrite(c_img_save_path, c_img)
    except():
        print("crop except")
        return


def generate_data(ori_img_path, save_path, template_match):
    """
    :param ori_img_path: 图片路径
    :param save_path: 保存路径
    :param template_match: 已经切割好的模板位置
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imgs = os.listdir(ori_img_path)
    for img in imgs:
        ori_img = cv2.imread(os.path.join(ori_img_path, img))
        g_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
        if img.split("_")[1] == "0.jpg":
            # ori_img, tempalte_size, save_path, seq, label):
            for t_img in template_match[:8]:
                crop_img(g_img, t_img, save_path, img.split("_")[0], "z")
        else:
            for t_img in template_match[8:]:
                crop_img(g_img, t_img, save_path, img.split("_")[0], "f")


def match_img(ori_img, template, thr_value):
    """
    # 增加一个模板匹配的校验 暂时没有用
    :param ori_img: 原始图片
    :param template: 模板图片
    :param thr_value: 匹配阈值
    :return:
    """
    g_ori_img = cv2.GaussianBlur(ori_img.copy(), ksize=(9, 7), sigmaX=0, sigmaY=0)
    g_template = cv2.GaussianBlur(template, ksize=(9, 7), sigmaX=0, sigmaY=0)
    res = cv2.matchTemplate(g_ori_img, g_template, cv2.TM_CCOEFF_NORMED)  # 模板匹配
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(max_val, max_loc)
    if thr_value <= max_val:
        print("匹配")


def find_zheng_and_fan_img(img_path, img, t_imgs, srcTri, save_path):
    """
    :param img_path:  图片路径地址
    :param img:  图片
    :param t_imgs:  模板列表
    :param srcTri:  映射的对应点
    :param save_path: 保存图片的路径
    :return:
    """

    image = cv2.imread(os.path.join(img_path, img), 0)  # 以灰度的形式读取图片
    # 进行模糊
    g_image = cv2.GaussianBlur(image.copy(), ksize=(9, 7), sigmaX=0, sigmaY=0)
    zheng_dstTri = []
    fan_dstTri = []
    # 进行正面的匹配
    for timg in t_imgs["zheng"]:
        res = cv2.matchTemplate(g_image, timg, cv2.TM_CCOEFF_NORMED)  # 模板匹配
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(max_val, max_loc)
        th, tw = timg.shape
        br = (max_loc[0] + tw, max_loc[1] + th)  # 右下点
        center = (max_loc[0] + tw / 2, max_loc[1] + th / 2)  # 中心点
        zheng_dstTri.append(center)
        # 画出矩形
        # cv2.rectangle(image, max_loc, br, (0, 0, 255), 2)
        # img_name = os.path.join(save_path, img)
        # cv2.imwrite(img_name, image)
    if len(zheng_dstTri) == 3:
        warp_mat = cv2.getAffineTransform(np.float32(zheng_dstTri), np.float32(srcTri["zheng"]))
        # mszie = cv2.imread(template_path, 0).shape

        # # cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
        zheng_result_img = cv2.warpAffine(image, warp_mat, (685, 436), borderMode=cv2.BORDER_REFLECT,
                                          borderValue=(255, 255, 255))
        cv2.imwrite(os.path.join(save_path, img.split(".")[0] + "_0.jpg"), zheng_result_img)
    else:
        print(f"正面只找到对应的模板的{len(zheng_dstTri)}个模板")

    # 进行反面的匹配
    for timg in t_imgs["fan"]:
        res = cv2.matchTemplate(g_image, timg, cv2.TM_CCOEFF_NORMED)  # 模板匹配
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(max_val, max_loc)
        th, tw = timg.shape
        br = (max_loc[0] + tw, max_loc[1] + th)  # 右下点
        center = (max_loc[0] + tw / 2, max_loc[1] + th / 2)  # 中心点
        fan_dstTri.append(center)
        # 画出矩形
        # cv2.rectangle(image, max_loc, br, (0, 0, 255), 2)
        # img_name = os.path.join(save_path, img)
        # cv2.imwrite(img_name, image)
    if len(fan_dstTri) == 3:
        warp_mat = cv2.getAffineTransform(np.float32(fan_dstTri), np.float32(srcTri["fan"]))
        # mszie = cv2.imread(template_path, 0).shape
        # # cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
        fan_result_img = cv2.warpAffine(image, warp_mat, (685, 436), borderMode=cv2.BORDER_REFLECT,
                                        borderValue=(255, 255, 255))
        cv2.imwrite(os.path.join(save_path, img.split(".")[0] + "_1.jpg"), fan_result_img)
    else:
        print(f"反面只找到对应的模板的{len(fan_dstTri)}个模板")

    return zheng_result_img, fan_result_img


def main(img_path, t_imgs, srcTri, save_path):
    imgs = os.listdir(img_path)
    # 找到图片
    for img in imgs:
        find_zheng_and_fan_img(img_path, img, t_imgs, srcTri, save_path)
    # 切割图片
    save_path = r"F:\PYcode\coding\MY_OCR_ID\cutimg"
    ori_img_path = r"F:\PYcode\coding\MY_OCR_ID\saveTem"
    generate_data(ori_img_path, save_path, template_match)


if __name__ == '__main__':
    """
    1. 找到正反面
    2. 切割正反面到一个一个的小块 
    3. 修改 地址的图片，修改成排字
    4. 
    """



