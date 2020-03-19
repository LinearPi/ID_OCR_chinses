# -*- coding: utf-8 -*-

# @Date    : 2020- 03- 05
# @Time    : 15: 57
# @Author  : Linear2pi

"""
1. 找到正反面
2. 切割正反面到一个一个的小块
3. 修改 地址的图片，修改成排字
4.
"""
import os

import cv2
import numpy as np

from find_split_img.cfg import srcTri, t_imgs, template_match


def change_img_size(image, save_change_img_path="", img=""):
    # 改变图片的大小只对 A4 纸张下复印的省份证图片
    if image.shape[1] < image.shape[0]:  # 根据原图的大小来确定图片是横着还是竖着
        print(image.shape[0], image.shape[1])
        image = cv2.resize(image, (1660, 2360))
    else:
        image = cv2.resize(image, (2360, 1660))
    if save_change_img_path and img:
        cv2.imwrite(os.path.join(save_change_img_path, img), image)
    return image


# 1.1对图片进行翻转
def match_img_result(ori_img, g_ori_img, g_template, thr_value, i=0):
    res = cv2.matchTemplate(g_ori_img, g_template, cv2.TM_CCOEFF_NORMED)  # 模板匹配
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(max_val, max_loc, "i:", i)
    if thr_value <= max_val:
        print("匹配")
        return True, ori_img
    ori_img = np.flip(ori_img, 0)
    g_ori_img = np.flip(g_ori_img, 0)
    i += 1
    print(i)
    if i > 3:
        return False, None
    # 递归调用自己方法
    return match_img_result(ori_img, g_ori_img, g_template, thr_value, i)


# 1.找到正反面 是否需要进行图片的翻转 正反面需要分开进行识别，有可能会出现图片是身份证的正反面是相反的情况
def match_img(ori_img, template, thr_value, i=0):
    """
    :param ori_img: 原始图片
    :param template: 模板图片
    :param thr_value: 匹配阈值
    :return: 翻转的图片
    """
    g_image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    g_ori_img = cv2.GaussianBlur(g_image, ksize=(9, 7), sigmaX=0, sigmaY=0)
    g_template = cv2.GaussianBlur(template, ksize=(9, 7), sigmaX=0, sigmaY=0)

    return match_img_result(ori_img, g_ori_img, g_template, thr_value)


def find_zheng_img(image, t_imgs, srcTri, img="", save_path=""):
    """
    :param image:
    :param img:
    :param t_imgs:
    :param srcTri:
    :param save_path:
    :return:
    """
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_image = cv2.GaussianBlur(g_image, ksize=(9, 7), sigmaX=0, sigmaY=0)
    zheng_dstTri = []
    # 进行正面的匹配
    for timg in t_imgs["zheng"]:
        res = cv2.matchTemplate(g_image, timg, cv2.TM_CCOEFF_NORMED)  # 模板匹配
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(max_val, max_loc)
        th, tw = timg.shape
        center = (max_loc[0] + tw / 2, max_loc[1] + th / 2)  # 中心点
        zheng_dstTri.append(center)

    if len(zheng_dstTri) == 3:
        warp_mat = cv2.getAffineTransform(np.float32(zheng_dstTri), np.float32(srcTri["zheng"]))
        zheng_result_img = cv2.warpAffine(image, warp_mat, (685, 436), borderMode=cv2.BORDER_REFLECT,
                                          borderValue=(255, 255, 255))
        cv2.imwrite(os.path.join(save_path, img.split(".")[0] + "_0.jpg"), zheng_result_img)
        return zheng_result_img
    else:
        print(f"正面只找到对应的模板的{len(zheng_dstTri)}个模板")
        return None


def find_fan_img(image, t_imgs, srcTri, img="", save_path=""):
    """
    :param image:
    :param img:
    :param t_imgs:
    :param srcTri:
    :param save_path:
    :return:
    """
    g_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g_image = cv2.GaussianBlur(g_image, ksize=(9, 7), sigmaX=0, sigmaY=0)
    # g_image = cv2.GaussianBlur(image.copy(), ksize=(9, 7), sigmaX=0, sigmaY=0)
    fan_dstTri = []
    # 进行反面的匹配
    for timg in t_imgs["fan"]:
        res = cv2.matchTemplate(g_image, timg, cv2.TM_CCOEFF_NORMED)  # 模板匹配
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print(max_val, max_loc)
        th, tw = timg.shape
        center = (max_loc[0] + tw / 2, max_loc[1] + th / 2)  # 中心点
        fan_dstTri.append(center)

    if len(fan_dstTri) == 3:
        warp_mat = cv2.getAffineTransform(np.float32(fan_dstTri), np.float32(srcTri["fan"]))  # 算出变换矩阵的系数
        fan_result_img = cv2.warpAffine(image, warp_mat, (685, 436), borderMode=cv2.BORDER_REFLECT,
                                        borderValue=(255, 255, 255))  # 三点影射
        cv2.imwrite(os.path.join(save_path, img.split(".")[0] + "_1.jpg"), fan_result_img)
        return fan_result_img
    else:
        print(f"反面只找到对应的模板的{len(fan_dstTri)}个模板")
        return None


# 2.1对单个图片进行切割
def crop_img(ori_img, tempalte_size, img, save_path=""):
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
        c_img_save_path = os.path.join(save_path, "%s_%s.jpg" % (img.split(".")[0], str(tempalte_size["index"])))
        if tempalte_size["index"] == 7:
            c_img = merge_address(c_img)
            c_img = detect_fn(c_img)

        if tempalte_size["index"] == 9:
            c_img = merge_issuing(c_img)
            c_img = detect_fn(c_img)
        if tempalte_size["index"] == 2:
            c_img = detect_fn(c_img)
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
        # #  进行二值化 阀值100
        # ret, c_img = cv2.threshold(c_img, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # 对图片进行视觉转化
        retina = cv2.bioinspired.Retina_create((c_img.shape[1], c_img.shape[0]))
        retina.run(c_img)
        # get our processed image :)
        c_img = retina.getParvo()
        cv2.imwrite(c_img_save_path, c_img)
    except():
        print("crop except")
        return


# 2. 对所有正反面进行切分
def crop_imgs(image, template_match, img, save_path):
    """
    :param ori_img_path: 图片路径
    :param save_path: 保存路径
    :param template_match: 已经切割好的模板位置
    """
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # imgs = os.listdir(ori_img_path)
    # for img in imgs:
    #     ori_img = cv2.imread(os.path.join(ori_img_path, img))
    #     g_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2GRAY)
    #     if img.split("_")[1] == "0.jpg":
    #         # ori_img, tempalte_size, save_path, seq, label):
    #         for t_img in template_match[:8]:
    #             crop_img(g_img, t_img, save_path, img.split("_")[0], "z")
    #     else:
    #         for t_img in template_match[8:]:
    #             crop_img(g_img, t_img, save_path, img.split("_")[0], "f")
    for t_img in template_match:
        crop_img(image, t_img, img.split("_")[0], save_path)


# 3.1进行图片的分割
def merge_address(img):
    """
    :param img address图片
    描述: 三行的地址数据和转换成一行
    """
    points = [[(0, 0), (288, 31)], [(3, 31), (288, 62)], [(3, 62), (288, 93)]]
    img_count = len(points)
    # 根据切割点对图片进行切割和拼接
    image3 = np.hstack([img[points[0][0][1]:points[0][1][1], points[0][0][0]:points[0][1][0]],
                        img[points[1][0][1]:points[1][1][1], points[1][0][0]:points[1][1][0]]])
    image3 = np.hstack([image3, img[points[2][0][1]:points[2][1][1], points[2][0][0]:points[2][1][0]]])
    return image3


def merge_issuing(img):
    """
    :param img 签发机关
    描述: 两行的签发机关
    """
    points = [[(0, 0), (290, 27)], [(10, 27), (290, 54)]]
    img_count = len(points)
    # 根据切割点对图片进行切割和拼接
    image2 = np.hstack([img[points[0][0][1]:points[0][1][1], points[0][0][0]:points[0][1][0]],
                        img[points[1][0][1]:points[1][1][1], points[1][0][0]:points[1][1][0]]])
    return image2


# 改变图片的大小和增加一些路径
def preprocess_img(img):
    resize_img = cv2.resize(img, (int(2.0 * img.shape[1]), int(2.0 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    # 放大两倍，更容易识别
    resize_img = cv2.convertScaleAbs(resize_img, alpha=0.35, beta=20)
    resize_img = cv2.normalize(resize_img, dst=None, alpha=300, beta=10, norm_type=cv2.NORM_MINMAX)
    img_blurred = cv2.medianBlur(resize_img, 7)  # 中值滤波
    img_blurred = cv2.medianBlur(img_blurred, 3)
    # 这里面的几个参数，alpha，beta都可以调节，目前感觉效果还行，但是应该还可以调整地更好
    return img_blurred


# 3.2 切除图片多余白色的部分
def detect_fn(image):
    resize_img = cv2.resize(image, (int(2.0 * image.shape[1]), int(2.0 * image.shape[0])),
                            interpolation=cv2.INTER_CUBIC)
    img = preprocess_img(image)
    # cv2.imwrite(img_save_path + img_name + '_processed.jpg', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # 两个参数可调
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    # cv2.imwrite(img_save_path + img_name + '_dilation.jpg', dilation2)

    region = []
    #  查找轮廓
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
    #  筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if (area < 50):
            continue
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 筛选那些太细的矩形，留下扁的
        if 25 < height < 80 and width > 25 and height < width * 1.3:
            region.append(box)
    max_x = 0
    for box in region:  # 每个box是左下，左上，右上，右下坐标
        for box_p in box:
            if box_p[0] > max_x:
                max_x = box_p[0]
    h, w, c = resize_img.shape
    return resize_img[0:h, 0:min(max_x + 50, w)]


# 3.对地址图片进行剪切
def change_one_img_size(img_path, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for img in os.listdir(img_path):
        img_c = cv2.imread(os.path.join(img_path, img))
        if "7" in img.split("_")[2]:
            img_c = merge_address(img_c)
            img_sss = detect_fn(img_c, img, save_path)
            cv2.imwrite(os.path.join(save_path, img), img_sss)


# 全过程
def find_split_process(img_path, save_path, t_imgs, srcTri, template_match):
    """
    :param img_path:  图片地址
    :param save_path: 保存切割好的图片地址
    :param t_imgs: 用于匹配的模板
    :param srcTri: 模板的位置
    :param template_match: 模板切割成小图片
    :param fan: 反面的模板
    :param zheng: 正面的模板
    :return:
    """

    fan = r"F:\PYcode\coding\MY_OCR_ID\template\fan.jpg"
    zheng = r"F:\PYcode\coding\MY_OCR_ID\template\zheng.jpg"
    # TODO 分隔路径
    two_save_path = os.path.join(os.path.dirname(save_path), "two_imgs")
    print(two_save_path)
    if not os.path.exists(two_save_path):
        os.mkdir(two_save_path)
    fan = cv2.imread(fan, 0)
    zheng = cv2.imread(zheng, 0)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    imgs = os.listdir(img_path)
    for img in imgs:
        image = cv2.imread(os.path.join(img_path, img), 1)
        # 1. 改变图片的大小到指定的A4大小
        ori_image = change_img_size(image)
        # cv2.imwrite(r"F:\PYcode\coding\OCR-Id\data_temp\chage.jpg", ori_image)
        flag, image = match_img(ori_image, zheng, 0.4)
        # cv2.imwrite(r"F:\PYcode\coding\OCR-Id\data_temp\match.jpg", image)
        # 2.匹配模板
        if flag:
            # image, t_imgs, srcTri, img="", save_path=""
            zheng_image = find_zheng_img(image, t_imgs, srcTri, img=img, save_path=two_save_path)
            # cv2.imwrite(r"F:\PYcode\coding\OCR-Id\data_temp\find.jpg", image)
            # 把图片切割成小块的 image, template_match, img, save_path
            crop_imgs(zheng_image, template_match[:8], img, save_path)

            fan_image = find_fan_img(image, t_imgs, srcTri,img=img, save_path=two_save_path)
            crop_imgs(fan_image, template_match[8:], img, save_path)
        else:
            print("没有匹配到对应的模板，请重新传照片")
            continue


if __name__ == '__main__':
    img_path = r"F:\PYcode\coding\OCR-Id\data\minidata"
    save_path = r"F:\PYcode\coding\OCR-Id\data\save_split"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    find_split_process(img_path, save_path, t_imgs, srcTri, template_match)
