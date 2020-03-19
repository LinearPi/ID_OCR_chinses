# -*- coding: utf-8 -*-

# @Date    : 2020- 03- 05
# @Time    : 15: 29
# @Author  : Linear2pi
import cv2
fan_srcTri = [(103, 110), (417, 110), (218, 348)]
zheng_srcTri = [(80, 103), (130, 369), (275, 155)]
srcTri = {"zheng": zheng_srcTri, "fan": fan_srcTri}
# 反面进行操作

fan = r"find_split_img/template/fan.jpg"
zheng = r"find_split_img/template/zheng.jpg"

guihui = r"find_split_img/template/fan_guohui.jpg"
fazheng = r"find_split_img/template/fan_fazhengjiguan.jpg"
renmin = r"find_split_img/template/fan_renmin.jpg"

zheng_chu = r"find_split_img/template/zheng_chu.jpg"
zheng_minzu = r"find_split_img/template/zheng_minzu.jpg"
zheng_gong = r"find_split_img/template/zheng_gong.jpg"
zheng_xing = r"find_split_img/template/zheng_xing.jpg"
sfhao = r"find_split_img/template/sfhao.jpg"


fgimg = cv2.imread(guihui, 0)
fgimg = cv2.GaussianBlur(fgimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

fimg = cv2.imread(fazheng, 0)
fimg = cv2.GaussianBlur(fimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

rimg = cv2.imread(renmin, 0)
rimg = cv2.GaussianBlur(rimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

fan_imgs = [fgimg, rimg, fimg]
cimg = cv2.imread(zheng_chu, 0)  # B
cimg = cv2.GaussianBlur(cimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

mimg = cv2.imread(zheng_minzu, 0)  # D
mimg = cv2.GaussianBlur(mimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

gimg = cv2.imread(zheng_gong, 0)  # C
gimg = cv2.GaussianBlur(gimg, ksize=(9, 7), sigmaX=0, sigmaY=0)

ximg = cv2.imread(zheng_xing, 0)  # A
ximg = cv2.GaussianBlur(ximg, ksize=(9, 7), sigmaX=0, sigmaY=0)
zheng_imgs = [ximg, gimg, mimg]

t_imgs = {"zheng": zheng_imgs,
          "fan": fan_imgs}

# 身份证上面各个元素的准确坐标,长宽,序号,用于从图片企鹅个
issuing_unit = {
    "x_d": 272,
    "y_d": 308,
    "w": 290,
    "h": 54,
    "index": 9}

effective_data = {
    "x_d": 272,
    "y_d": 363,
    "w": 290,
    "h": 29,
    "index": 10}

name = {
    "x_d": 124,
    "y_d": 56,
    "w": 98,
    "h": 33,
    "index": 1}

nationality = {
    "x_d": 262,
    "y_d": 116,
    "w": 80,
    "h": 28,
    "index": 2}

gender = {
    "x_d": 128,
    "y_d": 116,
    "w": 31,
    "h": 28,
    "index": 3}

birthday_year = {
    "x_d": 128,
    "y_d": 168,
    "w": 66,
    "h": 25,
    "index": 4}

birthday_month = {
    "x_d": 232,
    "y_d": 168,
    "w": 30,
    "h": 25,
    "index": 5}

birthday_day = {
    "x_d": 292,
    "y_d": 168,
    "w": 36,
    "h": 25,
    "index": 6}

address = {
    "x_d": 128,
    "y_d": 220,
    "w": 288,
    "h": 93,
    "index": 7}

id_card = {
    "x_d": 232,
    "y_d": 352,
    "w": 380,
    "h": 30,
    "index": 8}

template_match = [name, nationality, gender, birthday_year, birthday_month,
                  birthday_day, address, id_card, issuing_unit, effective_data]

