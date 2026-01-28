import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # To display the minus sign correctly
def main():
    dataset_name = 'tsmc'
    # 数据准备
    k_list = [50, 100, 200, 400]
    
    # 各项指标数据
    # cwq_Jan08-tkg/ cwq数据集在tkg上召回的结果
    # ans_recall = [0.765, 0.839, 0.897, 0.947]
    # shortest_path_quadruple_recall = [0.616, 0.706, 0.795, 0.873]
    # gpt_quadruple_recall = [0.636, 0.716, 0.795, 0.868]

    # cwq_Jan08-tkg/ tsmc数据集在tkg上召回的结果
    ans_recall = [0.452, 0.560, 0.643, 0.685]
    shortest_path_quadruple_recall = [0.353, 0.451, 0.540, 0.587]
    gpt_quadruple_recall = [0.373, 0.420, 0.486, 0.538]


    # 设置柱状图的宽度
    bar_width = 0.25
    
    # 设置X轴的位置
    r1 = np.arange(len(k_list))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # 创建画布
    plt.figure(figsize=(12, 7))
    
    # 绘制三组柱状图
    # 使用比较柔和的配色方案
    bars1 = plt.bar(r1, ans_recall, color='#4e79a7', width=bar_width, edgecolor='white', label='答案四元组召回率')
    bars2 = plt.bar(r2, shortest_path_quadruple_recall, color='#f28e2b', width=bar_width, edgecolor='white', label='最短路径四元组召回率')
    bars3 = plt.bar(r3, gpt_quadruple_recall, color='#e15759', width=bar_width, edgecolor='white', label='GPT四元组召回率')
    
    # 添加X轴标签和刻度
    plt.xlabel('K', fontweight='bold', fontsize=12)
    plt.xticks([r + bar_width for r in range(len(k_list))], k_list, fontsize=11)
    
    # 添加Y轴标签
    plt.ylabel('Recall', fontweight='bold', fontsize=12)
    plt.ylim(0, 1.15)  # 设置Y轴范围，留出顶部空间显示数值和图例
    
    # 添加标题
    # plt.title('CWQ Dataset Recall Metrics', fontweight='bold', fontsize=14)
    
    # 添加图例
    plt.legend(loc='upper left', fontsize=10)
    
    # 添加网格线（仅Y轴）
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 在柱状图上方添加数值标签的辅助函数
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=9, rotation=0)

    # 为每组柱子添加标签
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # 调整布局并保存
    plt.tight_layout()
    output_path = f'results/plots/{dataset_name}/{dataset_name}_quadruple_recall_metrics.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")

if __name__ == "__main__":
    main()