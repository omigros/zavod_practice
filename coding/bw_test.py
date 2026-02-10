import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ---------- 1D каскадное ядро ----------
def build_kernel(size):
    if size % 2 == 0 or size < 3:
        raise ValueError("Апертура должна быть нечётной и ≥ 3")

    k = np.array([1], dtype=float)

    # расширяем пока не достигнем размера
    while len(k) < size:
        k = np.convolve(k, [1, 1])

    # центрируем если перебор
    if len(k) > size:
        extra = len(k) - size
        k = k[extra//2 : extra//2 + size]

    return k


# ---------- построение 2D маски ----------
def build_filter(Ax, Ay):

    Rx = build_kernel(Ax)
    Ry = build_kernel(Ay)

    M = -np.outer(Ry, Rx)

    # корректировка центра
    s = -np.sum(M)

    corr = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ], dtype=float)

    corr *= s / np.sum(corr)

    cy = Ay//2
    cx = Ax//2

    M[cy-1:cy+2, cx-1:cx+2] += corr

    return M, Rx, Ry


# ---------- сепарабельная свёртка ----------
def separable_convolution(img, Rx, Ry):

    h, w = img.shape
    pad_x = len(Rx)//2
    pad_y = len(Ry)//2

    padded = np.pad(img, ((pad_y,pad_y),(pad_x,pad_x)), mode='reflect')

    temp = np.zeros_like(padded, dtype=float)

    # по строкам
    for y in range(pad_y, h+pad_y):
        for x in range(pad_x, w+pad_x):
            temp[y,x] = np.sum(
                padded[y, x-pad_x:x+pad_x+1] * Rx
            )

    out = np.zeros_like(img, dtype=float)

    # по столбцам
    for y in range(pad_y, h+pad_y):
        for x in range(pad_x, w+pad_x):
            out[y-pad_y, x-pad_x] = np.sum(
                temp[y-pad_y:y+pad_y+1, x] * Ry
            )

    return out


# ---------- основной запуск ----------
Ax = int(input("Апертура по X: "))
Ay = int(input("Апертура по Y: "))

mask, Rx, Ry = build_filter(Ax, Ay)

print("\nМаска фильтра:\n", mask)
print("\nСумма коэффициентов:", np.sum(mask))

img = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE).astype(float)

result = separable_convolution(img, Rx, Ry)

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Исходное")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Фильтр")
plt.imshow(result, cmap='gray')
plt.axis('off')

plt.show()