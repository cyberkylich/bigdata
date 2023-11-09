import matplotlib.pyplot as plt
import numpy as np

garage = np.array([100, 82, 105, 89, 102])
street = np.array([80, 98, 75, 91, 78])
print(f"Коэффициент корреляции {np.corrcoef(street, garage)[0, 1]}")

plt.grid(True)
plt.title("Диаграмма рассеяния", fontsize=20)
plt.xlabel("Кол-во припаркованных автомобилей в гараже")
plt.ylabel("Кол-во припаркованных автомобилей на улице")
plt.scatter(garage, street, marker='o', color="red")
plt.show()
