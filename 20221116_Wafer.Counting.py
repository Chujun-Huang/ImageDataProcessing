import cv2
import numpy as np
import os.path
import sys

# Read the .png file in grey level by opencv
Input = sys.argv[1]
print(f"{os.path.basename(Input)}")
IMG = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)
ori_rows, ori_cols = IMG.shape
print(f"Image size {ori_rows} x {ori_cols}")
# Calculated the image size
# ref: https://blog.gtwang.org/programming/opencv-basic-image-read-and-write-tutorial/
#num_rows, num_cols = IMG.shape
#print(num_rows, num_cols)

# First Count
T_IMG = IMG[0:ori_rows, 300:400]
num_rows, num_cols = T_IMG.shape
print(f"Sample size {num_rows} x {num_cols}")
#cv2.imwrite('test.png', T_IMG)

# Convolution
# ref: https://stackoverflow.com/questions/71805460/improving-a-median-filter-to-process-images-with-heavy-impulse-saltpepper-noi
# ref: https://setosa.io/ev/image-kernels/
Con_IMG = np.zeros(shape=(num_rows, num_cols))
Smooth_h = np.array([[1, 1, 1], [2, 2, 2], [1, 1, 1]])
Sharp = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
Sharp2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
IMG_0padding = np.pad(T_IMG, 1, mode='constant')
WindowSize = 3
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding[i:i+WindowSize, j:j+WindowSize]
        Smoothing = Window * Smooth_h
        Con_IMG[i, j] = np.sum(Smoothing) / 12
#cv2.imwrite('test_smooth.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp
        Con_IMG[i, j] = np.sum(Sharpping)
#cv2.imwrite('test_sharp_Sobel.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp2
        Con_IMG[i, j] = np.sum(Sharpping)
Con_IMG = np.clip(Con_IMG, 0, 255)
Con_IMG = Con_IMG ** 2 * 0.1
#cv2.imwrite('test_sharp_Sobel.sharp.png', Con_IMG)
Con_IMG[Con_IMG >= 50] = 255
Con_IMG[Con_IMG < 50] = 0
#cv2.imwrite('test_Binary_Sobel.sharp.png', Con_IMG)

# Count lines
CountLine_index = []
for lines in range(num_rows):
    Sample_sum = np.sum(Con_IMG[lines])
    Tolerance_mis = 5
    Threshold = 255 * (num_cols - Tolerance_mis)
    if Sample_sum >= Threshold:
        CountLine_index.append(1)
    else:
        CountLine_index.append(0)
CountLine_index[len(CountLine_index) - 1] = 0
CountLine_index[1] = 0
CountLine_index[2] = 0

# Remove redundant
CountLine_index_R = CountLine_index[::-1]
for num in range(len(CountLine_index_R)):
    if CountLine_index_R[num] == 1:
        if sum(CountLine_index_R[num:num+15]) != 1:
            CountLine_index_R[num] = 0
        else:
            pass
    else:
        pass

CountLine_index = CountLine_index_R[::-1]
FirstCounting_index = CountLine_index
FirstCounting_result = sum(CountLine_index)

# Second count
Middle = round(ori_cols / 2)
T_IMG = IMG[0:ori_rows, Middle:Middle+100]
IMG_0padding = np.pad(T_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding[i:i+WindowSize, j:j+WindowSize]
        Smoothing = Window * Smooth_h
        Con_IMG[i, j] = np.sum(Smoothing) / 12
#cv2.imwrite('test_smooth.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp
        Con_IMG[i, j] = np.sum(Sharpping)
#cv2.imwrite('test_sharp_Sobel.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp2
        Con_IMG[i, j] = np.sum(Sharpping)
Con_IMG = np.clip(Con_IMG, 0, 255)
Con_IMG = Con_IMG ** 2 * 0.1
#cv2.imwrite('test_sharp_Sobel.sharp.png', Con_IMG)
Con_IMG[Con_IMG >= 50] = 255
Con_IMG[Con_IMG < 50] = 0
#cv2.imwrite('test_Binary_Sobel.sharp.png', Con_IMG)

# Count lines
CountLine_index = []
for lines in range(num_rows):
    Sample_sum = np.sum(Con_IMG[lines])
    Tolerance_mis = 5
    Threshold = 255 * (num_cols - Tolerance_mis)
    if Sample_sum >= Threshold:
        CountLine_index.append(1)
    else:
        CountLine_index.append(0)
CountLine_index[len(CountLine_index) - 1] = 0
CountLine_index[1] = 0
CountLine_index[2] = 0

# Remove redundant
CountLine_index_R = CountLine_index[::-1]
for num in range(len(CountLine_index_R)):
    if CountLine_index_R[num] == 1:
        if sum(CountLine_index_R[num:num+15]) != 1:
            CountLine_index_R[num] = 0
        else:
            pass
    else:
        pass

CountLine_index = CountLine_index_R[::-1]
SecondCounting_index = CountLine_index
SecondCounting_result = sum(CountLine_index)

# Third count
T_IMG = IMG[0:ori_rows, ori_cols-400:ori_cols-300]
IMG_0padding = np.pad(T_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding[i:i+WindowSize, j:j+WindowSize]
        Smoothing = Window * Smooth_h
        Con_IMG[i, j] = np.sum(Smoothing) / 12
#cv2.imwrite('test_smooth.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp
        Con_IMG[i, j] = np.sum(Sharpping)
#cv2.imwrite('test_sharp_Sobel.png', Con_IMG)
IMG_0padding1 = np.pad(Con_IMG, 1, mode='constant')
for i in range(num_rows):
    for j in range(num_cols):
        Window = IMG_0padding1[i:i+WindowSize, j:j+WindowSize]
        Sharpping = Window * Sharp2
        Con_IMG[i, j] = np.sum(Sharpping)
Con_IMG = np.clip(Con_IMG, 0, 255)
Con_IMG = Con_IMG ** 2 * 0.1
#cv2.imwrite('test_sharp_Sobel.sharp.png', Con_IMG)
Con_IMG[Con_IMG >= 50] = 255
Con_IMG[Con_IMG < 50] = 0
#cv2.imwrite('test_Binary_Sobel.sharp.png', Con_IMG)

# Count lines
CountLine_index = []
for lines in range(num_rows):
    Sample_sum = np.sum(Con_IMG[lines])
    Tolerance_mis = 5
    Threshold = 255 * (num_cols - Tolerance_mis)
    if Sample_sum >= Threshold:
        CountLine_index.append(1)
    else:
        CountLine_index.append(0)
CountLine_index[len(CountLine_index) - 1] = 0
CountLine_index[1] = 0
CountLine_index[2] = 0

# Remove redundant
CountLine_index_R = CountLine_index[::-1]
for num in range(len(CountLine_index_R)):
    if CountLine_index_R[num] == 1:
        if sum(CountLine_index_R[num:num+15]) != 1:
            CountLine_index_R[num] = 0
        else:
            pass
    else:
        pass

CountLine_index = CountLine_index_R[::-1]
ThirdCounting_index = CountLine_index
ThirdCounting_result = sum(CountLine_index)

# Comparing counting result
IMG_C = cv2.imread(Input, cv2.IMREAD_COLOR)
if FirstCounting_result == SecondCounting_result:
    if FirstCounting_result == ThirdCounting_result:
        print(f"Counting result: {SecondCounting_result -1}")
        for num in range(len(SecondCounting_index)):
            if SecondCounting_index[num] == 1:
                cv2.line(IMG_C, (0, num), (IMG_C.shape[1], num), (0, 0, 255), 3)
        else:
            pass
    else:
        print(f"Counting result: {SecondCounting_result - 1}")
        for num in range(len(SecondCounting_index)):
            if SecondCounting_index[num] == 1:
                cv2.line(IMG_C, (0, num), (IMG_C.shape[1], num), (0, 0, 255), 3)
        else:
            pass
else:
    if FirstCounting_result == ThirdCounting_result:
        print(f"Counting result: {FirstCounting_result - 1}")
        for num in range(len(FirstCounting_index)):
            if FirstCounting_index[num] == 1:
                cv2.line(IMG_C, (0, num), (IMG_C.shape[1], num), (0, 0, 255), 3)
        else:
            pass
    else:
        print(f"Counting result: {SecondCounting_result - 1}")
        for num in range(len(SecondCounting_index)):
            if SecondCounting_index[num] == 1:
                cv2.line(IMG_C, (0, num), (IMG_C.shape[1], num), (0, 0, 255), 3)
        else:
            pass
# Output the result image with red lines
Basename = os.path.basename(Input)
Output = "Result_" + Basename
cv2.imwrite(Output, IMG_C)