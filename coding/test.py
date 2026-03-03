import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Системные настройки
# ------------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')
np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ------------------------------------------------------------
# Быстрые рециркуляторы
# ------------------------------------------------------------
def SR_fast(img, M):
    cs = np.cumsum(img, axis=1)
    out = cs.copy()
    out[:, M:] -= cs[:, :-M]
    return out

def KR_fast(img, M):
    cs = np.cumsum(img, axis=0)
    out = cs.copy()
    out[M:, :] -= cs[:-M, :]
    return out

def cascade(img, Mx, My):
    return KR_fast(SR_fast(img, Mx), My)

def CHRSF_lowpass(img, Mx, My):
    out = img.astype(np.float64)
    for _ in range(4):
        out = cascade(out, Mx, My)
    return out

# ------------------------------------------------------------
# Аналитические маски ЧРСФ
# ------------------------------------------------------------
def build_1d_mask(M):
    k = np.array([1.0])
    for _ in range(4):
        k = np.convolve(k, np.ones(M))
    return k

def build_masks(Ax_req, Ay_req):

    # параметры каскадов
    Mx = (Ax_req - 1) // 4 + 1
    My = (Ay_req - 1) // 4 + 1

    # реальная апертура ЧРСФ
    Ax_real = 4 * (Mx - 1) + 1
    Ay_real = 4 * (My - 1) + 1

    Rx = build_1d_mask(Mx)
    Ry = build_1d_mask(My)

    base = np.outer(Ry, Rx)

    final = -base.copy()

    sum_base = np.sum(base)

    cy = final.shape[0] // 2
    cx = final.shape[1] // 2

    # центральная компенсация
    final[cy, cx] += sum_base

    # коэффициент масштабирования относительно 3x3
    C = np.array([[1,2,1],
                  [2,4,2],
                  [1,2,1]], dtype=np.float64)

    k_corr = sum_base / np.sum(C)

    return base, final, k_corr, Ax_real, Ay_real, Mx, My

# ------------------------------------------------------------
# Повышение чёткости
# ------------------------------------------------------------
def CHRSF_sharpen(img, Ax_req, Ay_req, alpha):

    Mx = (Ax_req - 1) // 4 + 1
    My = (Ay_req - 1) // 4 + 1

    low = CHRSF_lowpass(img, Mx, My)

    low *= np.mean(img) / (np.mean(low) + 1e-9)

    high = img - low
    result = img + alpha * high

    return np.clip(result, 0, 255).astype(np.uint8)

# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------
print("=== ЧЕТЫРЁХКАСКАДНЫЙ РЕКУРСИВНО-СЕПАРАБЕЛЬНЫЙ ФИЛЬТР ===")

Ax_req = int(input("Введите Ax (нечётная): "))
Ay_req = int(input("Введите Ay (нечётная): "))
alpha  = float(input("Введите коэффициент усиления: "))

base_mask, final_mask, k_corr, Ax_real, Ay_real, Mx, My = build_masks(Ax_req, Ay_req)

print("\nЗаданная апертура: {} x {}".format(Ax_req, Ay_req))
print("Реальная апертура ЧРСФ: {} x {}".format(Ax_real, Ay_real))
print("Параметры каскадов: Mx = {}, My = {}".format(Mx, My))
print("Коэффициент масштабирования k = {:.6f}".format(k_corr))
print("Сумма финальной маски = {:.6f}".format(np.sum(final_mask)))

# ------------------------------------------------------------
# Загрузка изображения
# ------------------------------------------------------------
img = cv2.imread("lena.png", cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError("Файл lena.png не найден")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ------------------------------------------------------------
# Фильтрация
# ------------------------------------------------------------
start = time.perf_counter()

result = np.zeros_like(img)
for c in range(3):
    result[:,:,c] = CHRSF_sharpen(
        img[:,:,c],
        Ax_req,
        Ay_req,
        alpha
    )

end = time.perf_counter()

print("Время обработки: {:.4f} сек".format(end - start))

# ------------------------------------------------------------
# Сохранение (ИМЕННО ВВЕДЁННЫЕ Ax Ay)
# ------------------------------------------------------------
filename = f"result_{Ax_req}_{Ay_req}.png"
cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
print("Изображение сохранено как", filename)

# ------------------------------------------------------------
# Визуализация
# ------------------------------------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Исходное")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(result)
plt.title("После ЧРСФ")
plt.axis("off")

plt.show()