#!/usr/bin/env python3
# coding: utf-8
from dataclasses import dataclass, field
from datetime import datetime
from glob import iglob
from math import pi

import cv2
import numpy as np
import numpy.typing as npt
from utils.trackbar import AllIntTrackbar, FloatTrackbar, OddIntTrackbar, TwinAllIntTrackbar, TwinFloatTrackbar


@dataclass
class Image:
    input: np.ndarray
    name: str
    modified: np.ndarray = field(default=False)
    mask: np.ndarray = field(default=False)


def set_colors(images: npt.ArrayLike) -> npt.ArrayLike:
    """
    Range for HSV in openCV: H: 0-180°, S: 0-255, V: 0-255,
    Range for HLS in openCV: H: 0-180°, L: 0-255, S: 0-255,
    Range for LAB in openCV: L: 0-255, A: 0-255, B: 0-255.
    """
    cv2.destroyAllWindows()
    window_name = 'Color Space Parameters'

    if images.modified.ndim == 3:
        src = images.modified
    else:
        print("A 3-channel image is required to change the color space")
        return images
    visualize(image=src, name=window_name)

    type_ = AllIntTrackbar(window=window_name, trackbar='Type: \n 0: HSV \n 1: HLS \n 2: LAB',
                           value_min=0, value_max=2)
    first = TwinAllIntTrackbar(window=window_name, trackbar1='lower H/L*', trackbar2='higher H/L*',
                               value_min=0, value_max=255)
    second = TwinAllIntTrackbar(window=window_name, trackbar1='lower S/L/a*', trackbar2='higher S/L/a*',
                                value_min=0, value_max=255)
    third = TwinAllIntTrackbar(window=window_name, trackbar1='lower V/S/b*', trackbar2='higher V/S/b*',
                               value_min=0, value_max=255)

    while True:
        if type_.value == 0:
            image = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2HSV)
        elif type_.value == 1:
            image = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2HLS)
        else:
            image = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2LAB)

        mask = cv2.inRange(src=image,
                           lowerb=(first.value1, second.value1, third.value1),
                           upperb=(first.value2, second.value2, third.value2))

        result = cv2.bitwise_and(src1=src, src2=src, mask=mask)

        visualize(image=result, name=window_name)
        visualize(image=mask, name='Mask')

        k = cv2.waitKey(1) & 0xFF
        if k == 0x1B:  # esc: exit without new image
            break
        elif k == 0xD:  # enter: exit with new image
            images.modified = result
            images.mask = mask
            break
    cv2.destroyAllWindows()
    return images


def detect_blobs(images: npt.ArrayLike) -> npt.ArrayLike:
    """
    Area: Extracted blobs have an area between minArea (inclusive) and
          maxArea (exclusive).
    Color: This filter compares the intensity of a binary image at the
           center of a blob to blobColor. If they differ, the blob is
           filtered out. Use blobColor = 0 to extract dark blobs and
           blobColor = 255 to extract light blobs.
    Circularity: Extracted blobs have circularity (4*pi*A) / (d^2) with
                 (A=Area and d=perimeter) between minCircularity (inclusive)
                 and maxCircularity (exclusive).
    Ratio of inertia: Extracted blobs have this ratio between
                      minInertiaRatio (inclusive) and
                      maxInertiaRatio (exclusive).
    Convexity: Extracted blobs have convexity
               (area / area of blob convex hull) between minConvexity
               (inclusive) and maxConvexity (exclusive).
    """
    cv2.destroyAllWindows()
    window_name = 'Blob Detector Parameters'

    src = images.modified
    visualize(image=src, name=window_name)

    thresh = TwinAllIntTrackbar(window=window_name,
                                trackbar1='min Threshold',
                                trackbar2='max Threshold',
                                value_min=0, value_max=5000)
    thresh_step = AllIntTrackbar(window=window_name, trackbar='Threshold step', value_min=1, value_max=50)
    min_dist = AllIntTrackbar(window=window_name, trackbar='min Distance\nbetween Blobs',
                              value_min=10, value_max=500)
    min_repeat = AllIntTrackbar(window=window_name, trackbar='min Detections\nfor each Blob',
                                value_min=1, value_max=25)
    area = TwinAllIntTrackbar(window=window_name, trackbar1='min Area', trackbar2='max Area',
                              value_min=50, value_max=5000)
    filter_color = AllIntTrackbar(window=window_name, trackbar='Filter by Color', value_min=0, value_max=1)
    color = AllIntTrackbar(window=window_name, trackbar='Color:\n 0: dark, 255: bright', value_min=0, value_max=255)
    filter_circularity = AllIntTrackbar(window=window_name, trackbar='Filter by Circularity',
                                        value_min=0, value_max=1)
    circularity = TwinFloatTrackbar(window=window_name, trackbar1='min Circularity',
                                    trackbar2='max Circularity', limits=[0, 100, 0.0, 1.0])
    filter_convexity = AllIntTrackbar(window=window_name, trackbar='Filter by Convexity',
                                      value_min=0, value_max=1)
    convexity = TwinFloatTrackbar(window=window_name, trackbar1='min Convexity',
                                  trackbar2='max Convexity', limits=[0, 100, 0.0, 1.0])
    filter_inertia = AllIntTrackbar(window=window_name, trackbar='Filter by Inertia',
                                    value_min=0, value_max=1)
    inertia = TwinFloatTrackbar(window=window_name, trackbar1='min Inertia',
                                trackbar2='max Inertia', limits=[0, 100, 0.0, 1.0])

    result_mask = False

    while True:
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = thresh.value1
        params.maxThreshold = thresh.value2
        params.thresholdStep = thresh_step.value
        params.minDistBetweenBlobs = min_dist.value
        params.minRepeatability = min_repeat.value
        params.minArea = area.value1
        params.maxArea = area.value2
        params.filterByColor = filter_color.value
        params.blobColor = color.value
        params.filterByCircularity = filter_circularity.value
        params.minCircularity = circularity.value_interp1
        params.maxCircularity = circularity.value_interp2
        params.filterByConvexity = filter_convexity.value
        params.minConvexity = convexity.value1
        params.maxConvexity = convexity.value2
        params.filterByInertia = filter_inertia.value
        params.minInertiaRatio = inertia.value1
        params.maxInertiaRatio = inertia.value2

        detector = cv2.SimpleBlobDetector_create(params)

        kpts = detector.detect(src)
        pts = [point.pt for point in kpts]
        print(pts)
        result = cv2.drawKeypoints(src, kpts, (), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # the image is decoupled from the window due to the many trackbars
        visualize(image=np.zeros([1, 1, 3], dtype=np.uint8), name=window_name)
        visualize(image=result, name='Modified')
        if images.mask is not False:
            kpts = detector.detect(images.mask)
            result_mask = cv2.drawKeypoints(images.mask, kpts, (), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            visualize(image=result_mask, name='Mask')
        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            images.mask = result_mask
            break
    cv2.destroyAllWindows()
    return images


def detect_circles(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Circle Detector Parameters'

    src = images.modified
    visualize(image=src, name=window_name)

    min_distance = AllIntTrackbar(window=window_name, trackbar='min Distance', value_min=100, value_max=2500)
    param = TwinAllIntTrackbar(window=window_name, trackbar1='Parameter 1', trackbar2='Parameter 2',
                               value_min=5, value_max=500)
    radius = TwinAllIntTrackbar(window=window_name, trackbar1='min Radius', trackbar2='max Radius',
                                value_min=10, value_max=5000)

    result_mask = False

    while True:
        result = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(image=result, method=cv2.HOUGH_GRADIENT, dp=1,
                                   minDist=min_distance.value, param1=param.value1, param2=param.value2,
                                   minRadius=radius.value1, maxRadius=radius.value2)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(result, center, 1, (0, 0, 255), 3)
                # circle outline
                radius_ = i[2]
                cv2.circle(result, center, radius_, (0, 0, 255), 3)
        visualize(image=result, name=window_name)

        if images.mask is not False:
            result_mask = images.mask
            circles = cv2.HoughCircles(image=result_mask, method=cv2.HOUGH_GRADIENT, dp=1,
                                       minDist=min_distance.value, param1=param.value1, param2=param.value2,
                                       minRadius=radius.value1, maxRadius=radius.value2)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle center
                    cv2.circle(result_mask, center, 1, (0, 0, 255), 3)
                    # circle outline
                    radius_ = i[2]
                    cv2.circle(result_mask, center, radius_, (0, 0, 255), 3)
            visualize(image=result_mask, name='Mask')

        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            images.mask = result_mask
            break
    cv2.destroyAllWindows()
    return images


def adaptive_thresholding(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Adaptive Thresholding'

    src = images.modified
    visualize(image=src, name=window_name)

    type_ = AllIntTrackbar(window=window_name, trackbar='Type: \n 0: Binary \n 1: Binary Inverted',
                           value_min=0, value_max=1)
    method = AllIntTrackbar(window=window_name, trackbar='AdaptiveMethod: \n 0: Mean \n 1: Gaussian',
                            value_min=0, value_max=1)
    block_size = OddIntTrackbar(window=window_name, trackbar='Block size', value_min=3, value_max=255)
    constant = FloatTrackbar(window=window_name, trackbar='Constant C', limits=[0, 200, -15.0, 15.0])

    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    while True:
        result = cv2.adaptiveThreshold(
            src=src,
            maxValue=255,
            adaptiveMethod=method.value,
            thresholdType=type_.value,
            blockSize=block_size.value,
            C=constant.value_interp,
        )
        visualize(image=result, name=window_name)

        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit without new image
            break
        elif k == 0xD:  # enter: exit with new image
            images.modified = result
            break
    cv2.destroyAllWindows()
    return images


def thresholding(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Global Thresholding'

    src = images.modified
    visualize(image=src, name=window_name)

    type_ = AllIntTrackbar(window=window_name,
                           trackbar=('Type: \n 0: Binary \n 1: Binary Inverted \n '
                                     '2: Truncate \n 3: To Zero \n 4: To Zero Inverted'),
                           value_min=0, value_max=4)
    thresh = AllIntTrackbar(window=window_name, trackbar='Value', value_min=0, value_max=255)

    if src.ndim == 3:
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    while True:
        _, result = cv2.threshold(src, thresh.value, maxval=255, type=type_.value)
        visualize(image=result, name=window_name)

        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit without new image
            break
        elif k == 0xD:  # enter: exit with new image
            images.modified = result
            break
    cv2.destroyAllWindows()
    return images


def cut_filename(filename) -> str:
    return filename.split('/')[2].split('.')[0]


def rescale(image: npt.ArrayLike, scale: int) -> npt.ArrayLike:
    height, width, _ = image.shape
    width, height = int(width * scale / 100), int(height * scale / 100)
    return cv2.resize(src=image, dsize=(width, height))


def visualize(image, name: str, width: int = 800, height: int = 640):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, width=width, height=height)
    cv2.imshow(name, image)


def normalize_coordinates(col: int, row: int, image: npt.ArrayLike) -> tuple[float, float]:
    num_rows, num_cols = image.shape[:2]
    return col / (num_cols - 1.), row / (num_rows - 1.)


def denormalize_coordinates(x: float, y: float, image: npt.ArrayLike) -> tuple[int, int]:
    num_rows, num_cols = image.shape[:2]
    return int(x * (num_cols - 1.)), int(y * (num_rows - 1.))


def select_roi(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Select ROI'

    visualize(image=images.modified, name=window_name)
    x, y, w, h = cv2.selectROI(windowName=window_name, img=images.modified)
    xn, yn = normalize_coordinates(col=x, row=y, image=images.modified)
    wn, hn = normalize_coordinates(col=w, row=h, image=images.modified)
    print(f"x: {xn},\ny: {yn},\nw: {wn},\nh: {hn}")
    xd, yd = denormalize_coordinates(x=xn, y=yn, image=images.modified)
    tn = 5
    cv2.rectangle(img=images.input, pt1=(x-tn, y-tn), pt2=(x+w+tn, y+h+tn), color=(0, 0, 200), thickness=tn)
    cv2.destroyWindow(winname=window_name)
    images.modified = images.modified[int(y):int(y+h), int(x):int(x+w)]
    if images.mask is not False:
        images.mask = images.mask[int(y):int(y+h), int(x):int(x+w)]
    return images


def contrast(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Contrast'

    src = images.modified
    visualize(image=src, name=window_name)

    clip_limit = AllIntTrackbar(window=window_name, trackbar='Clip limit', value_min=1, value_max=50)
    grid_size = AllIntTrackbar(window=window_name, trackbar='Grid size', value_min=1, value_max=50)

    lab = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2LAB)
    l_src, a, b = cv2.split(lab)

    while True:
        clahe = cv2.createCLAHE(clipLimit=clip_limit.value, tileGridSize=(grid_size.value, grid_size.value))
        l1 = clahe.apply(l_src)
        result = cv2.merge((l1, a, b))
        result = cv2.cvtColor(src=result, code=cv2.COLOR_LAB2BGR)
        visualize(image=result, name=window_name)

        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            break
    cv2.destroyAllWindows()
    return images


def morph(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Morphological Transformation'

    src = images.modified
    visualize(image=src, name=window_name)

    operator = AllIntTrackbar(window=window_name,
                              trackbar='Operator: \n 0: Opening \n 1: Closing \n '
                                       '2: Gradient \n 3: Top Hat \n 4: Black Hat \n 5: Erode \n 6: Dilate',
                              value_min=0, value_max=6)
    element = AllIntTrackbar(window=window_name, trackbar='Element: \n 0: Rect \n 1: Cross \n 2: Ellipse',
                             value_min=0, value_max=2)
    kernel_size = AllIntTrackbar(window=window_name, trackbar='Kernel size', value_min=1, value_max=50)

    while True:
        kernel = cv2.getStructuringElement(
            shape=element.value,
            ksize=(kernel_size.value, kernel_size.value)
        )
        if operator.value < 5:
            result = cv2.morphologyEx(src=src, op=operator.value + 2, kernel=kernel)
        elif operator.value == 5:
            result = cv2.erode(src=src, kernel=kernel)
        else:
            result = cv2.dilate(src=src, kernel=kernel)

        visualize(image=result, name=window_name)
        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            break
    cv2.destroyAllWindows()
    return images


def smoothing(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Smoothing'

    src = images.modified
    visualize(image=src, name=window_name)

    operator = AllIntTrackbar(
        window=window_name,
        trackbar='Operator: \n 0: filter2D \n 1: blur \n 2: GaussianBlur \n 3: medianBlur \n 4: bilateralFilter',
        value_min=0, value_max=4)
    kernel_size = OddIntTrackbar(window=window_name, trackbar='Kernel size', value_min=1, value_max=29)

    if images.mask is not False:
        mask = images.mask
    else:
        mask = False
    result_mask = False

    while True:
        if operator.value == 0:
            kernel = np.ones((kernel_size.value, kernel_size.value), dtype=np.float32)
            kernel /= (kernel_size.value ** 2)
            result = cv2.filter2D(src=src, ddepth=-1, kernel=kernel)
            if mask is not False:
                result_mask = cv2.filter2D(src=mask, ddepth=-1, kernel=kernel)
        elif operator.value == 1:
            result = cv2.blur(src=src, ksize=(kernel_size.value, kernel_size.value))
            if mask is not False:
                result_mask = cv2.blur(src=mask, ksize=(kernel_size.value, kernel_size.value))
        elif operator.value == 2:
            result = cv2.GaussianBlur(src=src, ksize=(kernel_size.value, kernel_size.value), sigmaX=0)
            if mask is not False:
                result_mask = cv2.GaussianBlur(src=mask, ksize=(kernel_size.value, kernel_size.value), sigmaX=0)
        elif operator.value == 3:
            result = cv2.medianBlur(src=src, ksize=kernel_size.value)
            if mask is not False:
                result_mask = cv2.medianBlur(src=mask, ksize=kernel_size.value)
        else:
            result = cv2.bilateralFilter(src=src, d=kernel_size.value,
                                         sigmaColor=kernel_size.value * 2,
                                         sigmaSpace=kernel_size.value / 2)
            if mask is not False:
                result_mask = cv2.bilateralFilter(src=mask, d=kernel_size.value,
                                                  sigmaColor=kernel_size.value * 2,
                                                  sigmaSpace=kernel_size.value / 2)
        visualize(image=result, name=window_name)
        if mask is not False:
            visualize(image=result_mask, name='Mask')
        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            images.mask = result_mask
            break
    cv2.destroyAllWindows()
    return images


def canny_edge(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    window_name = 'Canny Edge Detection'

    src = images.modified
    visualize(image=src, name=window_name)

    thresh1 = AllIntTrackbar(window=window_name, trackbar='1. Threshold', value_min=0, value_max=5000)
    thresh2 = AllIntTrackbar(window=window_name, trackbar='2. Threshold', value_min=0, value_max=5000)
    aperture = OddIntTrackbar(window=window_name, trackbar='Aperture size for\nSobel Operator:\n3, 5 or 7',
                              value_min=3, value_max=7)

    if images.mask is not False:
        mask = images.mask
    else:
        mask = False
    result_mask = False

    while True:
        result = cv2.Canny(image=src, threshold1=thresh1.value, threshold2=thresh2.value, apertureSize=aperture.value)
        visualize(image=result, name=window_name)
        if mask is not False:
            result_mask = cv2.Canny(image=mask,
                                    threshold1=thresh1.value,
                                    threshold2=thresh2.value,
                                    apertureSize=aperture.value)
            visualize(image=result_mask, name='Mask')
        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
        elif k == 0xD:  # enter: exit blob detector with new image
            images.modified = result
            images.mask = result_mask
            break
    cv2.destroyAllWindows()
    return images


def find_contours(images: npt.ArrayLike) -> npt.ArrayLike:
    # TODO: def WIP
    cv2.destroyAllWindows()
    window_name = 'Find Contours'

    src = images.input

    while True:
        contours, hierarchy = cv2.findContours(image=src, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros((src.shape[0], src.shape[1], 3), dtype=np.uint8)
        for i, _ in enumerate(contours):
            color = (0, 0, 255)
            cv2.drawContours(image=result, contours=contours, contourIdx=i, color=color, thickness=2, lineType=cv2.LINE_8, hierarchy=hierarchy, maxLevel=0)

        visualize(result, window_name)
        k = cv2.waitKey(1)
        if k == 0x1B:  # esc: exit blob detector and without new image
            break
    return images


def clear(images: npt.ArrayLike) -> npt.ArrayLike:
    cv2.destroyAllWindows()
    images.modified = images.input
    images.mask = False
    return images


def save(images: npt.ArrayLike) -> None:
    date_time = datetime.now().strftime('%d%m%Y-%H%M%S')
    name = f'images/output/{images.name}_modified_{date_time}.png'

    cv2.imwrite(name,
                images.modified,
                params=[cv2.IMWRITE_PNG_COMPRESSION, 1])
    print(f'{name} saved')
    if images.mask is not False:
        name = f'images/output/{images.name}_mask_{date_time}.png'
        cv2.imwrite(name,
                    images.mask,
                    params=[cv2.IMWRITE_PNG_COMPRESSION, 1])
        print(f'{name} saved')


def object_detector(images: npt.ArrayLike) -> npt.ArrayLike:
    """Order of preprocessing:
        1. select ROI
        2. reduce noise (smoothing, morph, etc.)
        3. mask (thresholding of any kind)
        4. find objects (blob detector, canny edge, contours, etc.)"""
    cv2.destroyAllWindows()
    window_name = 'object Detector'

    #images = select_roi(images)
    src = images.modified

    x, y = denormalize_coordinates(x=0.5010053006762932, y=0.5415409925966548, image=src)
    w, h = denormalize_coordinates(x=0.034363004935112414, y=0.047162051000822595, image=src)
    tn = 5
    cv2.rectangle(img=src, pt1=(x-tn, y-tn), pt2=(x+w+tn, y+h+tn), color=(0, 0, 200), thickness=tn)
    src = src[int(y):int(y+h), int(x):int(x+w)]

    visualize(image=images.input, name='Input')
    visualize(image=src, name='Output')
    empty_img = np.zeros([1, 1, 3])
    visualize(image=empty_img, name=window_name)

    kernel_size = OddIntTrackbar(window=window_name, trackbar='Kernel size', value_min=1, value_max=39)
    sigma_color = AllIntTrackbar(window=window_name, trackbar='SigmaColor', value_min=1, value_max=250)
    sigma_space = AllIntTrackbar(window=window_name, trackbar='SigmaSpace', value_min=1, value_max=250)

    type_ = AllIntTrackbar(window=window_name,
                           trackbar=('Type: \n 0: Binary \n 1: Binary Inverted \n '
                                     '2: Truncate \n 3: To Zero \n 4: To Zero Inverted'),
                           value_min=0, value_max=4)
    thresh = AllIntTrackbar(window=window_name, trackbar='Value', value_min=0, value_max=255)

    thresh_blob = TwinAllIntTrackbar(window=window_name,
                                     trackbar1='min Threshold',
                                     trackbar2='max Threshold',
                                     value_min=0, value_max=5000)
    thresh_step = AllIntTrackbar(window=window_name, trackbar='Threshold step', value_min=1, value_max=50)
    min_dist = AllIntTrackbar(window=window_name, trackbar='min Distance\nbetween Blobs',
                              value_min=10, value_max=500)
    min_repeat = AllIntTrackbar(window=window_name, trackbar='min Detections\nfor each Blob',
                                value_min=1, value_max=25)
    area = TwinAllIntTrackbar(window=window_name, trackbar1='min Area', trackbar2='max Area',
                              value_min=2000, value_max=5000)
    filter_color = AllIntTrackbar(window=window_name, trackbar='Filter by Color', value_min=0, value_max=1)
    color = AllIntTrackbar(window=window_name, trackbar='Color:\n 0: dark, 255: bright', value_min=0, value_max=255)
    filter_circularity = AllIntTrackbar(window=window_name, trackbar='Filter by Circularity',
                                        value_min=0, value_max=1)
    circularity = TwinFloatTrackbar(window=window_name, trackbar1='min Circularity',
                                    trackbar2='max Circularity', limits=[0, 100, 0.0, 1.0])
    filter_convexity = AllIntTrackbar(window=window_name, trackbar='Filter by Convexity',
                                      value_min=0, value_max=1)
    convexity = TwinFloatTrackbar(window=window_name, trackbar1='min Convexity',
                                  trackbar2='max Convexity', limits=[0, 100, 0.0, 1.0])
    filter_inertia = AllIntTrackbar(window=window_name, trackbar='Filter by Inertia',
                                    value_min=0, value_max=1)
    inertia = TwinFloatTrackbar(window=window_name, trackbar1='min Inertia',
                                trackbar2='max Inertia', limits=[0, 100, 0.0, 1.0])

    cv2.setTrackbarPos(trackbarname='Kernel size', winname='object Detector', pos=3)
    cv2.setTrackbarPos(trackbarname='SigmaColor', winname='object Detector', pos=140)
    cv2.setTrackbarPos(trackbarname='SigmaSpace', winname='object Detector', pos=140)
    cv2.setTrackbarPos(trackbarname='Value', winname='object Detector', pos=150)
    #cv2.setTrackbarPos(trackbarname='Filter by Color', winname='object Detector', pos=0)
    cv2.setTrackbarPos(trackbarname='Filter by Circularity', winname='object Detector', pos=1)
    cv2.setTrackbarPos(trackbarname='min Circularity', winname='object Detector', pos=80)
    cv2.setTrackbarPos(trackbarname='max Circularity', winname='object Detector', pos=90)
    cv2.setTrackbarPos(trackbarname='max Circularity', winname='object Detector', pos=100)

    while True:
        # smoothing
        result = cv2.bilateralFilter(src=src, d=kernel_size.value,
                                     #sigmaColor=kernel_size.value * 2,
                                     sigmaColor=sigma_color.value,
                                     #sigmaSpace=kernel_size.value / 2)
                                     sigmaSpace=sigma_space.value)
        visualize(image=result, name='blurred')


        # thresholding
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, result = cv2.threshold(result, thresh.value, maxval=255, type=type_.value)

        # find object
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = thresh_blob.value1
        params.maxThreshold = thresh_blob.value2
        params.thresholdStep = thresh_step.value
        params.minDistBetweenBlobs = min_dist.value
        params.minRepeatability = min_repeat.value
        params.minArea = area.value1
        params.maxArea = area.value2
        params.filterByColor = filter_color.value
        params.blobColor = color.value
        params.filterByCircularity = filter_circularity.value
        params.minCircularity = circularity.value_interp1
        params.maxCircularity = circularity.value_interp2
        params.filterByConvexity = filter_convexity.value
        params.minConvexity = convexity.value1
        params.maxConvexity = convexity.value2
        params.filterByInertia = filter_inertia.value
        params.minInertiaRatio = inertia.value1
        params.maxInertiaRatio = inertia.value2

        detector = cv2.SimpleBlobDetector_create(params)

        kpts = detector.detect(result)
        # keypoints: size=diameter, angle=orientation,
        # response=known as the strength of the keypoint, it is the
        # keypoint detector response on the keypoint
        result = cv2.cvtColor(src=result, code=cv2.COLOR_GRAY2BGR)
        result = cv2.drawKeypoints(result, kpts, None, (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for keypoint in kpts:
            point = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            radius = int(keypoint.size / 2)
            area_ = pi * radius**2
            cv2.circle(img=result, center=point, radius=radius, color=(0, 0, 255), thickness=2)

        visualize(image=empty_img, name=window_name)
        visualize(image=result, name='Output')

        k = cv2.waitKey(40)
        if k == 0x1B:  # esc: exit without new image
            break
        elif k == 0xD:  # enter: exit with new image
            images.modified = result
            break
    cv2.destroyAllWindows()
    return images


def main(scale_percent: int = 100) -> None:
    for file_path in iglob(pathname='images/input/*.png', recursive=False):
        img_src = cv2.imread(filename=file_path)
        if scale_percent != 100:
            img_src = rescale(image=img_src, scale=scale_percent)
        images = Image(input=img_src, name=cut_filename(file_path), modified=img_src)
        while True:
            visualize(image=images.modified, name=f'Image: {cut_filename(file_path)}')
            if images.mask is not False:
                visualize(image=images.mask, name=f'Mask: {cut_filename(file_path)}')

            k = cv2.waitKey(1) & 0xFF
            if k == 0x31:  # 1: color space & color thresholding
                images = set_colors(images=images)
            elif k == 0x32:  # 2: smoothing
                images = smoothing(images=images)
            elif k == 0x33:  # 3: contrast
                images = contrast(images=images)
            elif k == 0x34:  # 4: global thresholding
                images = thresholding(images=images)
            elif k == 0x35:  # 5: adaptive thresholding
                images = adaptive_thresholding(images=images)
            elif k == 0x36:  # 6: morphological transformation
                images = morph(images=images)
            elif k == 0x37:  # 7: # find contours
                continue
                #images = find_contours(images=images)
            elif k == 0x38:  # 8: free
                continue
            elif k == 0x39:  # 9: free
                images = object_detector(images=images)
            elif k == 0x62:  # b: activate blob detector
                images = detect_blobs(images=images)
            elif k == 0x63:  # c: clean windows
                images = clear(images=images)
            elif k == 0x65:  # e: (canny) edge detection
                images = canny_edge(images=images)
            elif k == 0x1B:  # esc: exit the script
                exit(0)
            elif k == 0x68:  # h: activate hough circle (circle detector)
                images = detect_circles(images=images)
            elif k == 0x6E:  # n: next image
                break
            elif k == 0x72:  # r: select ROI
                images = select_roi(images=images)
            elif k == 0x73:  # s: save all current images
                save(images=images)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(scale_percent=100)
