import os
from glob import glob
import numpy as np
import cv2
from skimage import measure
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math

def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst

def normalize(src, img_size):
    dst = cv2.addWeighted(src, 4, cv2.GaussianBlur(src, (0, 0), img_size / 30), -4, 128)
    return dst

def remove_boundaries(src, img_size):
    mask = np.zeros(src.shape)
    cv2.circle(
        mask,
        center=(src.shape[1] // 2, src.shape[0] // 2),
        radius=int(img_size / 2 * 0.9),
        color=(1, 1, 1),
        thickness=-1)
    dst = src * mask + 128 * (1 - mask)
    return dst

def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

# Auto Cropping.
def eye_locate_find(img, minr, maxr):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=20, minRadius=minr, maxRadius=maxr)
    if circles is not None:
        return circles
    else:
        return []
def count_morethan_right_thresh(right_y_array_range, counts_right, thre_rt):
    count = 0
    temp = 0
    for i in range(len(right_y_array_range)):
        if count > 15:
            break
        if right_y_array_range[i] > np.argmax(counts_right):
            count += 1
            temp = i
        else:
            temp = 0
            count = 0

    if temp > 0:
        return True
    else:
        return False
def count_morethan_left_thresh(left_y_array_range, counts_left, thre_lt):
    count = 0
    temp = 0
    for i in range(len(left_y_array_range)):
        if count > 15:
            break
        if left_y_array_range[i] > np.argmax(counts_left):
            count += 1
            temp = i
        else:
            temp = 0
            count = 0
    if temp > 0:
        return True
    else:
        return False
def check_rectangle(img, top, down, left, right, times, thresh_right, thresh_left):
    check_pass = True
    early_stop_check = False
    h, w = img.shape
    search_range = int(h / (3 + times)) if times < 4 else int(h / 20)
    outerimage = img.copy()
    backgroundimage = img.copy()
    outerimage[top:down, left:right] = 0
    backgroundimage[top:down, left:right] = 0
    x_right = right
    y_right = int((top + down) / 2)
    right_limit = int(w) - right
    max_yindex_right = 0
    max_yindex_left = 0
    max_xindex_right = 0
    max_xindex_left = 0

    right_x_array = []
    right_y_array = []
    right_array_bg = []
    if right_limit != 0:
        for i in range(right_limit - 1):
            right_x_array.append(outerimage[y_right][x_right + 1 + i])
    for i in range(h - 1):
        r = x_right
        if x_right + 1 > w - 1:
            r = w - 1
        if i < h:
            right_y_array.append(outerimage[i][r])
    if len(right_x_array) < 1:
        right_array_bg = right_y_array
    else:
        right_array_bg = np.hstack((right_x_array, right_y_array))
    counts_right = np.bincount(right_array_bg)

    right_y_array_range = right_y_array[y_right - search_range:y_right + search_range]
    y_left = int((top + down) / 2)
    left_limit = 0 + left
    left_x_array = []
    left_y_array = []
    left_array_bg = []
    if left_limit != 0:
        for i in range(left_limit - 1):
            left_x_array.append(outerimage[y_left][left - (i + 1)])
    for i in range(h - 1):
        l = left - 1
        if left - 1 < 0:
            l = 0
        if i < h:
            left_y_array.append(outerimage[i][l])

    if len(left_x_array) < 1:
        left_array_bg = left_y_array
    else:
        left_array_bg = np.hstack((left_x_array, left_y_array))

    counts_left = np.bincount(left_array_bg)
    left_y_array_range = left_y_array[y_left - search_range:y_left + search_range]

    right_threshold = int((abs(thresh_right - np.argmax(counts_right)) / 2) - 10)
    left_threshold = int((abs(thresh_left - np.argmax(counts_left)) / 2) - 10)
    thre_rt = int(right_threshold) if right_threshold > 15 else 15
    thre_lt = int(left_threshold) if left_threshold > 15 else 15

    if np.argmax(counts_right) < thresh_right or np.argmax(counts_left) < thresh_left:
        # check_pass=True
        # thre_rt = 15 if times>1 else thre_rt
        # thre_lt = 15 if times>1 else thre_lt
        if np.max(right_y_array_range) - np.argmax(counts_right) > thre_rt or np.max(left_y_array_range) - np.argmax(
                counts_left) > thre_lt:
            isMore_right = count_morethan_right_thresh(right_y_array_range, counts_right, thre_rt)
            isMore_left = count_morethan_left_thresh(left_y_array_range, counts_left, thre_lt)
            if isMore_right == True or isMore_left == True:
                if (np.max(right_y_array_range) - np.argmax(counts_right) >= thre_rt) and isMore_right == True:
                    max_yindex_right = right_y_array.index(max(right_y_array_range))  # max index
                    max_xindex_right = right + 1
                if (np.max(left_y_array_range) - np.argmax(counts_left) >= thre_lt) and isMore_left == True:
                    max_yindex_left = left_y_array.index(max(left_y_array_range))  # max index
                    max_xindex_left = left - 1
                check_pass = False
            else:
                check_pass = True
        else:
            check_pass = True
    else:
        early_stop_check = True
        check_pass = False
    max_right_xarry = []
    max_left_xarry = []
    if right_limit != 0:
        for i in range(right_limit - 1):
            max_right_xarry.append(outerimage[max_yindex_right][max_xindex_right + i])

    if left_limit != 0:
        for j in range(left_limit - 1):
            max_left_xarry.append(outerimage[max_yindex_left][max_xindex_left - j])

    if np.sum(right_x_array) > np.sum(max_right_xarry):
        max_yindex_right = y_right
    if np.sum(left_x_array) > np.sum(max_left_xarry):
        max_yindex_left = y_left
    return check_pass, max_yindex_right, max_yindex_left, max_xindex_right, max_xindex_left, np.argmax(
        counts_right), np.argmax(counts_left), thre_rt, thre_lt, early_stop_check
def modify_rectangle(img, max_yindex_right, max_yindex_left, max_xindex_right, max_xindex_left, background_right,
                     background_left, thresh_right, thresh_left):
    outerimage = img.copy()
    h, w = img.shape
    right_limit = w - max_xindex_right - 1
    left_limit = 0 + max_xindex_left
    right_arr = []
    left_arr = []
    if max_xindex_right != 0:
        for i in range(right_limit):
            right_arr.append(outerimage[max_yindex_right][max_xindex_right + i])
    if max_xindex_left != 0:
        for j in range(left_limit):
            left_arr.append(outerimage[max_yindex_left][max_xindex_left - j])
    maxindex_xright = 0
    maxindex_xleft = 0
    accumulate_count = 5
    thresh_right = 15
    thresh_left = 15
    if len(right_arr) > 20 or len(left_arr) > 20:
        accumulate_count = 20
    stop_r_accumulate = 0
    stop_l_accumulate = 0

    no_record = 0
    temp_index = 0
    noise = False
    if len(right_arr) >= 5:
        for ii in range(len(right_arr) - 1):
            temp_max = int(right_arr[ii]) - int(right_arr[ii + 1])
            if temp_max > thresh_right and no_record == 0:
                no_record = 1
                temp_max = temp_max
                temp_index = ii
            if stop_r_accumulate < accumulate_count:
                if right_arr[ii] - background_right < int(thresh_right - 8):
                    maxindex_xright = ii
                    stop_r_accumulate += 1
                else:
                    stop_r_accumulate = 0
            if maxindex_xright == len(right_arr) - 2:
                maxindex_xright = temp_index
    temp_index_l = 0
    if len(left_arr) >= 5:
        for jj in range(len(left_arr) - 1):
            temp_max_l = int(left_arr[jj]) - int(left_arr[jj + 1])
            if temp_max_l > thresh_left and no_record == 0:
                no_record = 1
                temp_max_l = temp_max_l
                temp_index_l = jj
            if stop_l_accumulate < accumulate_count:
                if left_arr[jj] - background_left < int(thresh_left - 8):
                    maxindex_xleft = jj
                    stop_l_accumulate += 1
                else:
                    stop_l_accumulate = 0
            if maxindex_xleft == len(left_arr) - 2:
                maxindex_xleft = temp_index_l
    return maxindex_xright, maxindex_xleft
def cropping_final_checking(original_img, r, left_top, left_down, right_top, right_down):
    imgcopy = original_img.copy()
    height, width = imgcopy.shape
    left_top = left_top
    left_down = left_down
    right_top = right_top
    right_down = right_down
    thresh_right, binaryimg_right = cv2.threshold(imgcopy[0:height - 1, int(width / 2):width - 1], 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_left, binaryimg_left = cv2.threshold(imgcopy[0:height - 1, 0:int(width / 2)], 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    times = 1
    noise_case = False
    ischeckstatus_sucess = False
    early_stop_check = False
    check_status, max_yindex_right, max_yindex_left, max_xindex_right, max_xindex_left, background_right, background_left, thre_rt, thre_lt, early_stop_check = check_rectangle(
        original_img, left_top[1], left_down[1], left_top[0], right_down[0], times, thresh_right, thresh_left)
    ischeckstatus_sucess = check_status
    early_stop_check = early_stop_check
    if check_status == True:
        left_top = [left_top[0] - 5, left_top[1]]
        left_down = [left_down[0] - 5, left_down[1]]
        right_top = [right_top[0] + 5, right_top[1]]
        right_down = [right_down[0] + 5, right_down[1]]
    if check_status == False and early_stop_check == True:
        ischeckstatus_sucess = False
    if check_status == False and early_stop_check == False:
        for i in range(6):
            if ischeckstatus_sucess == False:
                times = i + 2
                maxindex_xright, maxindex_xleft = modify_rectangle(original_img, max_yindex_right, max_yindex_left,
                                                                   max_xindex_right, max_xindex_left, background_right,
                                                                   background_left, thre_rt, thre_lt)
                bias_right = 15 if maxindex_xright != 0 or times < 3 else 0
                bias_left = 15 if maxindex_xleft != 0 or times < 3 else 0
                maxindex_xleft = maxindex_xleft + bias_left
                maxindex_xright = maxindex_xright + bias_right
                left_top = [left_top[0] - maxindex_xleft, left_top[1]]
                left_down = [left_down[0] - maxindex_xleft, left_down[1]]
                right_top = [right_top[0] + maxindex_xright, right_top[1]]
                right_down = [right_down[0] + maxindex_xright, right_down[1]]

                check_status, max_yindex_right, max_yindex_left, max_xindex_right, max_xindex_left, background_right, background_left, thre_rt, thre_lt, early_stop_check = check_rectangle(
                    original_img, left_top[1], left_down[1], left_top[0], right_down[0], times, thresh_right,
                    thresh_left)
                ischeckstatus_sucess = check_status

    return ischeckstatus_sucess, left_top, left_down, right_top, right_down
class AutoCropping:
    def detect_image(self, image, color='RGB'):
        # self.name=name
        self.isDetectionSucess = False
        self.isCroppingCheckPass = False
        self.DetectionNoSucess_reason = ''
        self.noNeedCropping = False
        self.image = image
        self.center_x = 0
        self.center_y = 0
        self.center_r = 0
        self.color = color
        if self.color == 'RGB':
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.color == 'BGR':
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
        img = cv2.GaussianBlur(img, (15, 15), 0)
        h, w = img.shape
        if abs(w - h) > h / 4:
            self.noNeedCropping = False
            minRa = int(h / 2) - 20
            maxRa = int(w / 2)
            circles = eye_locate_find(img, minRa, maxRa)
            if len(circles) != 0:
                if len(circles[0]) > 1:
                    center_r = int(circles[0][0][2])
                    center_x = int(circles[0][0][0])
                    center_y = int(circles[0][0][1])
                    # L=0
                    x1 = np.array([circles[0][0][0], circles[0][0][1]])
                    for i in range(len(circles[0])):
                        circle_right_x = circles[0][i][0] + circles[0][i][2]
                        circle_left_x = circles[0][i][0] - circles[0][i][2]
                        if circle_right_x > w - 1:
                            circle_right_x = w - 1
                        if circle_left_x < 0:
                            circle_left_x = 0
                        if img[int(circles[0][i][1]), int(circle_right_x)] > 10 or img[
                            int(circles[0][i][1]), int(circle_right_x)] > 10:
                            if int(circles[0][i][2]) > center_r + 10:
                                x2 = np.array([circles[0][i][0], circles[0][i][1]])
                                L = np.linalg.norm(x1 - x2, ord=None, axis=None, keepdims=False)
                                if L < 30:
                                    center_r = circles[0][i][2]
                                    center_x = circles[0][i][0]
                                    center_y = circles[0][i][1]
                        if i > 2:
                            break
                else:
                    center_x = int(circles[0][0][0])
                    center_y = int(circles[0][0][1])
                    center_r = int(circles[0][0][2])
                real_range_w = [int(w / 2) - int(w / 5), int(w / 2) + int(w / 5)]
                real_range_h = [int(h / 2) - int(h / 5), int(h / 2) + int(h / 5)]
                if center_x >= real_range_w[0] and center_x <= real_range_w[1] and center_y >= real_range_h[
                    0] and center_y <= real_range_h[1]:
                    self.isDetectionSucess = True
                    self.center_x = center_x
                    self.center_y = center_y
                    self.center_r = center_r
                else:
                    self.isDetectionSucess = False
                    self.DetectionNoSucess_reason = 'out_of_normal_range'
            else:
                self.isDetectionSucess = False
                self.DetectionNoSucess_reason = 'no_eyes'
        else:
            self.noNeedCropping = True

    def detection_rectangle_coordinates(self):
        ##原圖大小切割位置
        original_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        if self.color == 'RGB':
            original_img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        elif self.color == 'BGR':
            original_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        original_img = cv2.GaussianBlur(original_img, (15, 15), 0)
        height, width = original_img.shape
        center_x = self.center_x
        center_y = self.center_y
        center_r = self.center_r
        a = int(center_x * 10)
        b = int(center_y * 10)
        r = int(center_r * 10)
        left_top = []
        right_top = []
        left_down = []
        right_down = []
        left_bias = 25
        right_bias = 25
        isPass = False

        if center_x != 0 and center_r != 0:
            if a - r - left_bias < 0:
                left_bias = 0
            if a + r + right_bias > width:
                right_bias = 0
            left = (a - r) - left_bias
            right = (a + r) + right_bias
            if left < 0:
                left = 0
            if right > width:
                right = width - 1
            left_top = [left, b - r]
            right_top = [right, b - r]
            left_down = [left, b + r]
            right_down = [right, b + r]
            if left_top[1] < 0:
                left_top = [left, 0]
                right_top = [right, 0]
            elif left_top[1] > 0:
                remaining = int((b - r) / 2.5)
                left_top = [left, b - r - remaining]
                right_top = [right, b - r - remaining]
                if b - r - remaining - 2 > 0:
                    if original_img[b - r - remaining - 2, int((left + right) / 2)] > int(
                            (original_img[b - r - remaining + 2, int((left + right) / 2)]) / 2):
                        left_top = [left, 0]
                        right_top = [right, 0]
                else:
                    left_top = [left, 0]
                    right_top = [right, 0]
            if left_down[1] < height:
                remaining = int((b - r) / 2.5)
                left_down = [left, b + r + remaining]
                right_down = [right, b + r + remaining]
                if b + r + remaining + 2 < height:
                    if original_img[b + r + remaining + 2, int((left + right) / 2)] > int(
                            (original_img[b + r + remaining - 2, int((left + right) / 2)]) / 2):
                        left_down = [left, height - 1]
                        right_down = [right, height - 1]
                else:
                    left_down = [left, height - 1]
                    right_down = [right, height - 1]
            elif left_down[1] > height:
                left_down = [left, height - 1]
                right_down = [right, height - 1]

            isPass, left_top1, left_down1, right_top1, right_down1 = cropping_final_checking(original_img, r, left_top,
                                                                                             left_down, right_top,
                                                                                             right_down)

            if isPass == True:
                self.isCroppingCheckPass = True
                if left_top1[0] < 0:
                    left_top1[0] = 0
                    left_down1[0] = 0
                if right_top1[0] > width - 1:
                    right_top1[0] = width - 1
                    right_down1[0] = width - 1
                left_top = [left_top1[0], left_top1[1]]
                left_down = [left_down1[0], left_down1[1]]
                right_top = [right_top1[0], right_top1[1]]
                right_down = [right_down1[0], right_down1[1]]

        return left_top, right_top, left_down, right_down

    # radius_range is the radius of circle for cutting , max is 1.0
    def get_circle(self, image, radius_range=0.9):
        if self.noNeedCropping == False:
            if self.isDetectionSucess == True:
                lt, rt, ld, rd = self.detection_rectangle_coordinates()
                if self.isCroppingCheckPass == True:
                    top = lt[1]
                    down = ld[1]
                    left = lt[0]
                    right = rd[0]
                    crop_img = image[top:down, left:right]
                    image = crop_img

        image_zero = np.zeros(image.shape)
        y, x, z = image.shape
        if y > x:
            radius = x
        else:
            radius = y
        radius = radius * 0.5
        cv2.circle(image_zero, (int(image.shape[1] / 2), int(image.shape[0] / 2)), int(radius * radius_range),
                   (1, 1, 1), -1, 8,
                   0)
        image = image * image_zero + 0 * (1 - image_zero)
        return image

def gamma_trans(img, gamma):  # gamma
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def nothing(x):
    pass

def preprocess(dataset, img_size, red_free=False,  scale=False, norm=False, pad=False, remove=False):
    if dataset == 'RNFL':
        df = pd.read_csv(Path("./data/dataset/train/label.csv"))
        img_paths = './data/dataset/train/' + df['id_code'].values + '.png'
        # img_paths = 'processed/train_images_resized/' + df['id_code'].values + '.png'
    elif dataset == 'diabetic_retinopathy':
        df = pd.read_csv('inputs/diabetic-retinopathy-resized/trainLabels.csv')
        img_paths = 'inputs/diabetic-retinopathy-resized/resized_train/' + df['image'].values + '.jpeg'
    elif dataset == 'test':
        df = pd.read_csv('./data/dataset/test/label.csv')
        img_paths = './data/dataset/test/' + df['id_code'].values + '.png'
    elif dataset == 'messidor':
        img_paths = glob('inputs/messidor/*/*.tif')
    else:
        NotImplementedError

    dir_name = 'processed/%s/images_%d' %(dataset, img_size)
    if scale:
        dir_name += '_scaled'
    if norm:
        dir_name += '_normed'
    if pad:
        dir_name += '_pad'
    if remove:
        dir_name += '_rm'
    if red_free:
        dir_name += '_redfree'

    os.makedirs(dir_name, exist_ok=True)
    for i in tqdm(range(len(img_paths))):
        img_path = img_paths[i]
        if os.path.exists(os.path.join(dir_name, os.path.basename(img_path))):
            continue
        img = cv2.imread(img_path)
        img_gray = cv2.imread(img_path, 0)
        try:
            if scale:
                img = scale_radius(img, img_size=img_size, padding=pad)
        except Exception as e:
            print(img_paths[i])
        img = cv2.resize(img, (img_size, img_size))
        if red_free:
            # img = np.fft.fft2(img)
            # img = np.fft.fftshift(img)
            # img = 20*np.log(np.abs(img))
            # b, img, r = cv2.split(lab)

            img = np.uint8(img)
            img_gray = np.uint8(img_gray)

            # Auto Gamma.
            mean = np.mean(img_gray)
            gamma_val = math.log10(0.4) / math.log10(mean / 255)
            img = gamma_trans(img, gamma_val)


            # 2 methods of red free.
            # b, g, r = cv2.split(img)
            # img = cv2.addWeighted(b, 0.5, g, 0.5, 0)
            img = img[:, :, 1]


            # # Auto Equal.
            img = cv2.GaussianBlur(img, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=2.25, tileGridSize=(5, 5))
            img = clahe.apply(img)
        if norm:
            img = normalize(img, img_size=img_size)
        if remove:
            img = remove_boundaries(img, img_size=img_size)
        cv2.imwrite(os.path.join(dir_name, os.path.basename(img_path)), img)

    return dir_name
