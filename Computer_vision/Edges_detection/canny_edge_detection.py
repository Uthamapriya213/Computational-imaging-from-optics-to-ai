import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32)

def normalize(img):
    return np.uint8(np.clip(img, 0, 255))


def laplacian_edge(img):
    lap = cv2.Laplacian(img, cv2.CV_32F,ksize=5)
    return lap

def log_edge(img, ksize=5, sigma=1):
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    log = cv2.Laplacian(blur, cv2.CV_64F,ksize =5)
    return log

def compute_gradient(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    mag = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)

    return mag, angle

def nms(mag, angle):
    H, W = mag.shape
    out = np.zeros((H, W))

    angle = angle * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):

            q = 0
            r = 0

            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]

            elif (22.5 <= angle[i,j] < 67.5):
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]

            elif (67.5 <= angle[i,j] < 112.5):
                q = mag[i+1, j]
                r = mag[i-1, j]

            elif (112.5 <= angle[i,j] < 157.5):
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i,j] >= q and mag[i,j] >= r:
                out[i,j] = mag[i,j]

    return out


def hysteresis(img, low, high):
    strong = 255
    weak = 75

    H, W = img.shape
    res = np.zeros_like(img)

    strong_pts = list(zip(*np.where(img >= high)))
    weak_pts = list(zip(*np.where((img >= low) & (img < high))))

    for i,j in strong_pts:
        res[i,j] = strong
    for i,j in weak_pts:
        res[i,j] = weak

    q = deque(strong_pts)

    neighbors = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),          (0,1),
        (1,-1),  (1,0),  (1,1)
    ]

    while q:
        i, j = q.popleft()

        for di, dj in neighbors:
            ni, nj = i+di, j+dj

            if 0 <= ni < H and 0 <= nj < W:
                if res[ni,nj] == weak:
                    res[ni,nj] = strong
                    q.append((ni,nj))

    res[res != strong] = 0
    return res
def canny_custom(img):
    blur = cv2.GaussianBlur(img, (5,5), 1)

    mag, angle = compute_gradient(blur)
    thin = nms(mag, angle)
    edges = hysteresis(thin, 30, 80)

    return edges

def canny_opencv(img):
    return cv2.Canny(img.astype(np.uint8),threshold1=30,threshold2=80)
img = load_gray("lena.jpeg")

lap = laplacian_edge(img)
log = log_edge(img)
canny_c = canny_custom(img)
canny_cv = canny_opencv(img)

plt.figure(figsize=(10,6))

plt.subplot(2,2,1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.subplot(2,2,2); plt.imshow(lap, cmap='gray'); plt.title("Laplacian")
plt.subplot(2,2,3); plt.imshow(log, cmap='gray'); plt.title("LoG")
plt.subplot(2,2,4); plt.imshow(canny_c, cmap='gray'); plt.title("Canny Custom")

plt.show()

