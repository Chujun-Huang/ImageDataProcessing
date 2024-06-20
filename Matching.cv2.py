import cv2
import numpy as np
import os.path
import sys

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

Source = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
Template_Border = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)
Template_Circle = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE)

# Border matching
Result = cv2.matchTemplate(Source, Template_Border, 3)
Nor_result = Result * 255
Nor_result_row, Nor_result_column = Nor_result.shape
Matching_coordinate = Nor_result.argmax()
Matching_column = Matching_coordinate % Nor_result_column
Matching_row = Matching_coordinate // Nor_result_column
IMG_Color = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
Border_label_img = label(IMG_Color, Matching_column, Matching_row, 300, (0, 0, 255))

# Circle / cross matching
Matching_img = np.zeros(shape=(300, 300))
for i in range(Matching_row, Matching_row + 300):
    for j in range(Matching_column, Matching_column + 300):
        row = i - Matching_row
        column = j - Matching_column
        Matching_img[column, row] = Source[i, j]
cv2.imwrite('temp.bmp', Matching_img)
Matching_border = cv2.imread("temp.bmp", cv2.IMREAD_GRAYSCALE)
Result_2 = cv2.matchTemplate(Matching_border, Template_Circle, 3)
Nor_result = Result_2 * 255
Nor_result_row, Nor_result_column = Nor_result.shape
Matching_coordinate = Nor_result.argmax()
Matching_column_2 = (Matching_coordinate % Nor_result_column) + Matching_column
Matching_row_2 = (Matching_coordinate // Nor_result_column) + Matching_row
print(Matching_column_2, Matching_row_2)
label_img = label(Border_label_img, Matching_column_2, Matching_row_2, 170, (0, 255, 0))
Basename = os.path.basename(sys.argv[1])
Output = "cv2.Result_" + Basename
cv2.imwrite(Output, label_img)