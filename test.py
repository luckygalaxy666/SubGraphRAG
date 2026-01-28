import matplotlib.pyplot as plt

# 1. 设置默认字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 2. 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 

# --- 测试代码 ---
plt.figure()
plt.plot([1, 2, 3], [-1, -2, -3])
plt.title("测试：中文标题与负号")
plt.xlabel("X轴")
plt.ylabel("Y轴")
plt.savefig("test.png")