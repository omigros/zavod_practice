import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import sys
import time
7
5
11
sys.stdout.reconfigure(encoding='utf-8')
np.set_printoptions(precision=6, suppress=True)


# ---------- 1D рекурсивно-каскадное ядро ----------
def build_kernel(size):

    if size % 2 == 0 or size < 3:
        raise ValueError("Апертура должна быть нечётной и ≥ 3")

    k = np.array([1.0], dtype=np.float64)

    while len(k) < size:
        k = np.convolve(k, [1, 1])

    if len(k) > size:
        d = len(k) - size
        k = k[d//2 : d//2 + size]

    return k


# ---------- формирование фильтра ----------
def build_filter(Ax, Ay):

    Rx = build_kernel(Ax)
    Ry = build_kernel(Ay)

    base = -np.outer(Ry, Rx)
    Sb = np.sum(base)

    C = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=np.float64)

    k = -Sb / np.sum(C)
    corr = k * C

    final = base.copy()
    cy, cx = Ay // 2, Ax // 2
    final[cy-1:cy+2, cx-1:cx+2] += corr

    energy = np.sum(np.abs(base))
    final /= energy

    return base, corr, final


# ---------- ввод параметров ----------
print("=== Рекурсивно-сепарабельный фильтр повышения чёткости ===")

Ax = int(input("Апертура X (нечётная): "))
Ay = int(input("Апертура Y (нечётная): "))

base, corr, final = build_filter(Ax, Ay)

print(f"\nФИНАЛЬНАЯ МАСКА ({Ay}x{Ax}):")
print(final)
print("Сумма:", np.sum(final))


# ---------- изображение ----------
img = cv2.imread("lena.png", cv2.IMREAD_UNCHANGED)
if img is None:
    raise FileNotFoundError("Файл lena.png не найден")

img = img.astype(np.float64)
img_blur = cv2.GaussianBlur(img, (3, 3), 0.5)

start = time.perf_counter()

if img.ndim == 2:
    filtered = cv2.filter2D(img_blur, -1, final)
else:
    filtered = np.zeros_like(img)
    for c in range(3):
        filtered[:, :, c] = cv2.filter2D(img_blur[:, :, c], -1, final)

end = time.perf_counter()
print(f"\nВремя обработки: {end - start:.4f} сек")


# ---------- визуализация ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.30)

if img.ndim == 3:
    im0 = ax[0].imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    im1 = ax[1].imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
else:
    im0 = ax[0].imshow(img.astype(np.uint8), cmap='gray')
    im1 = ax[1].imshow(img.astype(np.uint8), cmap='gray')

ax[0].set_title("Исходное изображение")
ax[1].set_title("После фильтрации")
for a in ax:
    a.axis('off')


# ---------- переменная хранения результата ----------
current_result = img.astype(np.uint8)


# ---------- ползунок силы эффекта ----------
ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
slider = Slider(ax_slider, "Сила эффекта", 0.0, 3.0, valinit=0.0)


def update(val):
    global current_result
    alpha = slider.val
    res = img + alpha * filtered
    res = np.clip(res, 0, 255).astype(np.uint8)
    current_result = res

    if img.ndim == 3:
        im1.set_data(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    else:
        im1.set_data(res)

    fig.canvas.draw_idle()


slider.on_changed(update)


# ---------- кнопка сохранения ----------
ax_button = plt.axes([0.4, 0.05, 0.2, 0.05])
btn = Button(ax_button, "Сохранить")


def save_image(event):
    filename = f"result_{Ax}_{Ay}.png"
    cv2.imwrite(filename, current_result)
    print(f"Изображение сохранено как {filename}")


btn.on_clicked(save_image)


# стартовое обновление
update(0.0)

plt.show()