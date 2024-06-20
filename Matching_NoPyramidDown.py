import cv2
import numpy as np
import math
import os.path
import sys

def pyramid_down(Source, Kernel):
    Kernel_H = len(Kernel)
    Kernel_W = len(Kernel[0])
    Sorurce_H, Source_W = Source.shape
    Source_0padding = np.pad(Source, 5, mode='constant')
    Pyramid_down_array = np.zeros(shape=(Sorurce_H // 2, Source_W // 2))
    for i in range(2, Sorurce_H, 2):
        for j in range(2, Source_W, 2):
            Window = Source_0padding[i:i+Kernel_H, j:j+Kernel_W]
            temp = Window * Kernel
            x = int(i / 2)
            y = int(j / 2)
            Pyramid_down_array[x, y] = np.sum(temp)
    Pyramid_down_array_H, Pyramid_down_array_W = Pyramid_down_array.shape
    print(f"Pyramid down H = {Pyramid_down_array_H}, W = {Pyramid_down_array_W}\n")
    return Pyramid_down_array

def matching(Start_site_y, Start_site_x, Source, Template):
    Source_H, Source_W = Source.shape
    Template_H, Template_W = Template.shape
    Template_var = np.var(Template)
    Template_mean = np.mean(Template)
    Template_zero_mean = Template - Template_mean
    Matching_img = np.zeros(shape=(Source_H -Template_H + 1, Source_W - Template_W + 1))
    for i in range(Start_site_x, Source_H - Template_H + 1):
        for j in range(Start_site_y, Source_W - Template_W + 1):
            Window = Source[i:i+Template_H, j:j+Template_W]
            Source_var = np.var(Window)
            Source_mean = np.mean(Window)
            Source_zero_mean = Window - Source_mean
            NCC = np.mean(Source_zero_mean * Template_zero_mean) / math.sqrt(Source_var * Template_var)
            Matching_img[i, j] = NCC
    Nor_Matching_IMG = (Matching_img + 1) * 255 / 2
    Nor_Matching_IMG_H, Nor_Matching_IMG_W = Nor_Matching_IMG.shape
    Matching_coordinate = Nor_Matching_IMG.argmax()
    Matching_x = Matching_coordinate % Nor_Matching_IMG_W
    Matching_y = Matching_coordinate // Nor_Matching_IMG_W
    return Nor_Matching_IMG, Matching_x, Matching_y

def matching_inside(Start_site_y, Start_site_x, End_site_y, End_site_x, Source, Template):
    Source_H, Source_W = Source.shape
    Template_H, Template_W = Template.shape
    Template_var = np.var(Template)
    Template_mean = np.mean(Template)
    Template_zero_mean = Template - Template_mean
    Matching_img = np.zeros(shape=(Source_H -Template_H + 1, Source_W - Template_W + 1))
    for i in range(Start_site_x, End_site_x):
        for j in range(Start_site_y, End_site_y):
            Window = Source[i:i+Template_H, j:j+Template_W]
            Source_var = np.var(Window)
            Source_mean = np.mean(Window)
            Source_zero_mean = Window - Source_mean
            NCC = np.mean(Source_zero_mean * Template_zero_mean) / math.sqrt(Source_var * Template_var)
            Matching_img[i, j] = NCC
    Nor_Matching_IMG = (Matching_img + 1) * 255 / 2
    Nor_Matching_IMG_H, Nor_Matching_IMG_W = Nor_Matching_IMG.shape
    Matching_coordinate = Nor_Matching_IMG.argmax()
    Matching_x = Matching_coordinate % Nor_Matching_IMG_W
    Matching_y = Matching_coordinate // Nor_Matching_IMG_W
    return Nor_Matching_IMG, Matching_x, Matching_y

def label(Input, Matching_x, Matching_y, Edge_length, Color):
    Source_color = Input
    Half_length = int(round(Edge_length / 2))
    cv2.line(Source_color, (Matching_x, Matching_y), (Matching_x + Edge_length, Matching_y), Color, 3)
    cv2.line(Source_color, (Matching_x, Matching_y), (Matching_x, Matching_y + Edge_length), Color, 3)
    cv2.line(Source_color, (Matching_x + Edge_length, Matching_y), (Matching_x + Edge_length, Matching_y + Edge_length), Color, 3)
    cv2.line(Source_color, (Matching_x, Matching_y + Edge_length), (Matching_x + Edge_length, Matching_y + Edge_length), Color, 3)
    cv2.line(Source_color, (Matching_x + Half_length, Matching_y), (Matching_x + Half_length, Matching_y + Edge_length), Color, 1)
    cv2.line(Source_color, (Matching_x, Matching_y + Half_length), (Matching_x + Edge_length, Matching_y + Half_length), Color, 1)
    return Source_color

# Read input
Source = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
Template_Border = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
Template_Circle = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)

# Calculate image size
Source_H, Source_W = Source.shape
print(f"Source image high: {Source_H}\nSource image width: {Source_W}")
print(f"")
Template_Border_H, Template_Border_W = Template_Border.shape
print(f"Template_Border high: {Template_Border_H}\nTemplate_Border width: {Template_Border_W}")
print(f"")
Template_Circle_H, Template_Circle_W = Template_Circle.shape
print(f"Template Circle high: {Template_Circle_H}\nTemplate Circle width: {Template_Circle_W}")
print(f"")

# Source image pyramid down
#Gaussian_kernel = ([[0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625], [0.015625, 0.0625, 0.09375, 0.0625, 0.015625], [0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375], [0.015625, 0.0625, 0.09375, 0.0625, 0.015625], [0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625]])
#Pyramid_down_Source = pyramid_down(Source, Gaussian_kernel)
#Pyramid_down_Template_Border = pyramid_down(Template_Border, Gaussian_kernel)

# Matching border in pyramid down image
#Matching_img, Matching_x, Matching_y = matching(0, 0, Pyramid_down_Source, Pyramid_down_Template_Border)
#print(Matching_x, Matching_y)
#cv2.imwrite('Matching_pyramid_down_border_test3.bmp', Matching_img)

# Matching the border in origin image size and label
Ori_matching_img, Ori_matching_x, Ori_matching_y = matching(0, 0, Source, Template_Border)
#cv2.imwrite('Matching.border_test4.bmp', Ori_matching_img)
Source_image_color = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
Border_label_img = label(Source_image_color, Ori_matching_x, Ori_matching_y, 300, (0, 0, 255))
#cv2.imwrite('Matching.Border_output3.bmp', Border_label_img)

# Matching the Circle in origin image and label
Cir_matching_img, Cir_matching_x, Cir_matching_y = matching_inside(Ori_matching_x, Ori_matching_y, Ori_matching_x + Template_Border_W - Template_Circle_W + 1, Ori_matching_y + Template_Border_H - Template_Circle_H + 1, Source, Template_Circle)
Circle_label_img = label(Border_label_img, Cir_matching_x, Cir_matching_y, 170, (0, 255, 0))
Basename = os.path.basename(sys.argv[1])
Output = "Result.NP_" + Basename
cv2.imwrite(Output, Circle_label_img)