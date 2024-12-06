import seaborn as sns
import matplotlib.pyplot as plt

# 获取 Set3 调色板中的颜色
colors = sns.color_palette("Set3", 7)

# 打印颜色
for i, color in enumerate(colors):
    print(f"Color {i+1}: {color}")

# 可视化颜色
sns.palplot(colors)
plt.show()