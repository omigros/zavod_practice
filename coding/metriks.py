import numpy as np
import cv2
import sys

sys.stdout.reconfigure(encoding='utf-8')
np.set_printoptions(precision=6, suppress=True)


# ---------- имя обработанного изображения ----------
result_filename = "result_11_7.png"   # ← менять вручную


# ---------- загрузка изображений ----------
original = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)
processed = cv2.imread(result_filename, cv2.IMREAD_GRAYSCALE)

if original is None:
    raise ValueError("Файл lena.png не найден")

if processed is None:
    raise ValueError(f"Файл {result_filename} не найден")

if original.shape != processed.shape:
    raise ValueError("Размеры изображений не совпадают")


# ---------- расчёт ----------
diff = original.astype(np.float32) - processed.astype(np.float32)

mse = np.mean(diff ** 2)
sko = np.sqrt(mse)

if mse == 0:
    possh = float('inf')
else:
    possh = 10 * np.log10((255 ** 2) / mse)


# ---------- вывод ----------
print("\nРезультаты обработки:")
print(f"СКО  = {sko:.6f}")
print(f"ПОСШ = {possh:.6f} дБ")