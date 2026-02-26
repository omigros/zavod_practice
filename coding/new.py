import numpy as np
import cv2
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ------------------------------------------------------------
# Системные настройки
# ------------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')
np.set_printoptions(precision=4, suppress=True, linewidth=140)

# ------------------------------------------------------------
# Быстрые рециркуляторы (сепарабельные)
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
    for _ in range(4):                     # ЧЕТЫРЕ КАСКАДА
        out = cascade(out, Mx, My)
    return out

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

    return np.clip(result, 0, 255).astype(np.uint8), Mx, My

# ------------------------------------------------------------
# Аналитические маски ЧРСФ
# ------------------------------------------------------------
def build_1d_mask(M):
    k = np.array([1.0])
    for _ in range(4):
        k = np.convolve(k, np.ones(M))
    return k

def build_masks(Ax_req, Ay_req):

    Mx = (Ax_req - 1) // 4 + 1
    My = (Ay_req - 1) // 4 + 1

    Ax_real = 4 * (Mx - 1) + 1
    Ay_real = 4 * (My - 1) + 1

    Rx = build_1d_mask(Mx)
    Ry = build_1d_mask(My)

    # ---------------- Основная маска ----------------
    base = np.outer(Ry, Rx)

    # ---------------- Корректирующая маска ----------------
    C = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=np.float64)

    k_corr = np.sum(base) / np.sum(C)
    corr = k_corr * C

    # ---------------- Финальная маска ----------------
    final = -base.copy()
    cy, cx = final.shape[0] // 2, final.shape[1] // 2
    final[cy-1:cy+2, cx-1:cx+2] += corr

    return Rx, Ry, base, corr, final, k_corr, Ax_real, Ay_real

# ------------------------------------------------------------
# Аккуратный вывод масок
# ------------------------------------------------------------
def print_mask(title, M, Ax_real, Ay_real):
    print(f"\n{title}")
    print(f"Апертура: Ax = {Ax_real}, Ay = {Ay_real}")
    print(f"Сумма коэффициентов: {np.sum(M):.6f}")
    print(M)

# ------------------------------------------------------------
# Основная программа
# ------------------------------------------------------------
print("=== Рекурсивно-сепарабельный ЧЕТЫРЁХКАСКАДНЫЙ фильтр ===")

Ax_req = int(input("Введите Ax (нечётная): "))
Ay_req = int(input("Введите Ay (нечётная): "))
alpha  = float(input("Введите коэффициент усиления: "))

Rx, Ry, base_mask, corr_mask, final_mask, k_corr, Ax_real, Ay_real = build_masks(Ax_req, Ay_req)

print("\n1D маска по X:")
print(Rx)

print("\n1D маска по Y:")
print(Ry)

print_mask("Основная 2D апертура", base_mask, Ax_real, Ay_real)

print("\nКорректирующая маска (3×3)")
print("Коэффициент масштабирования k =", f"{k_corr:.6f}")
print(corr_mask)

print_mask("Финальная маска повышения чёткости", final_mask, Ax_real, Ay_real)

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
    result[:, :, c], _, _ = CHRSF_sharpen(
        img[:, :, c], Ax_req, Ay_req, alpha
    )

end = time.perf_counter()
print(f"\nВремя обработки: {end - start:.4f} сек")

# ------------------------------------------------------------
# Визуализация + сохранение
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

ax[0].imshow(img)
ax[0].set_title("Исходное изображение")
ax[0].axis("off")

ax[1].imshow(result)
ax[1].set_title("После ЧРСФ")
ax[1].axis("off")

ax_btn = plt.axes([0.4, 0.1, 0.2, 0.08])
btn = Button(ax_btn, "Сохранить")

def save(event):
    filename = f"CHRSF_Ax{Ax_real}_Ay{Ay_real}.png"
    cv2.imwrite(filename, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Изображение сохранено как {filename}")

btn.on_clicked(save)

plt.show()