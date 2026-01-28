import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 设置中文字体，确保图表中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_time_entry(t_val):
    """
    尝试解析时间，支持 'YYYY-MM-DD' 字符串或年份数字/字符串。
    返回 datetime 对象，如果解析失败返回 None。
    """
    if pd.isna(t_val) or t_val == '':
        return None
    
    t_str = str(t_val).strip()
    
    # 尝试 YYYY-MM-DD 格式
    try:
        return datetime.strptime(t_str, '%Y-%m-%d')
    except ValueError:
        pass
    
    # 尝试纯年份 (例如 '2023', '2023.0')
    try:
        # 先转float处理 '2023.0' 这种情况，再转int
        year = int(float(t_str))
        # 构造该年1月1日
        return datetime(year, 1, 1)
    except ValueError:
        pass
        
    return None

def main():
    parser = argparse.ArgumentParser(description="统计CSV文件中四元组随时间的变化情况（次数与得分）")
    parser.add_argument('--csv_path', type=str, required=True, help="CSV文件路径")
    parser.add_argument('--time_col', type=str, default='time', help="时间列的列名，默认为 'time'")
    parser.add_argument('--score_col', type=str, default='score', help="得分列的列名，默认为 'score'")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"错误: 文件不存在 -> {args.csv_path}")
        return

    print(f"正在读取文件: {args.csv_path} ...")
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    # 检查列是否存在
    if args.time_col not in df.columns:
        print(f"错误: CSV中未找到时间列 '{args.time_col}'。现有列: {df.columns.tolist()}")
        return
        
    has_score = True
    if args.score_col not in df.columns:
        print(f"警告: CSV中未找到得分列 '{args.score_col}'。将只统计出现次数。")
        has_score = False

    # 解析时间
    print("正在解析时间数据...")
    df['parsed_datetime'] = df[args.time_col].apply(parse_time_entry)
    
    # 过滤无效时间
    valid_df = df.dropna(subset=['parsed_datetime']).copy()
    invalid_count = len(df) - len(valid_df)
    if invalid_count > 0:
        print(f"已过滤 {invalid_count} 条时间格式无效的数据。")
    
    if len(valid_df) == 0:
        print("没有有效的带时间数据，无法绘图。")
        return

    # 按时间聚合统计
    print("正在统计数据...")
    agg_dict = {'parsed_datetime': 'count'} # 用于计数
    if has_score:
        agg_dict[args.score_col] = 'mean'

    # 使用 groupby 进行聚合
    # 这里我们创建一个临时列用于计数
    valid_df['_count'] = 1
    stats = valid_df.groupby('parsed_datetime').agg({
        '_count': 'sum',
        **({args.score_col: 'mean'} if has_score else {})
    }).reset_index()
    
    stats = stats.sort_values('parsed_datetime')

    # 绘图
    print("正在生成图表...")
    output_dir = os.path.dirname(args.csv_path)
    if output_dir == '': output_dir = '.'
    filename_base = os.path.splitext(os.path.basename(args.csv_path))[0]
    
    # --- 图1：出现次数 ---
    plt.figure(figsize=(12, 6))
    plt.plot(stats['parsed_datetime'], stats['_count'], marker='.', linestyle='-', color='#1f77b4', label='出现次数')
    plt.ylabel('出现次数', fontsize=12, fontweight='bold')
    plt.title('四元组随时间变化：出现次数统计', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.xlabel('时间', fontsize=12, fontweight='bold')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    count_output_path = os.path.join(output_dir, f'{filename_base}_count_over_time.png')
    plt.savefig(count_output_path, dpi=300)
    print(f"出现次数图表已保存至: {count_output_path}")
    plt.close()

    # --- 图2：平均得分 (如果有) ---
    if has_score:
        plt.figure(figsize=(12, 6))
        plt.plot(stats['parsed_datetime'], stats[args.score_col], marker='.', linestyle='-', color='#d62728', label='平均得分')
        plt.ylabel('平均得分', fontsize=12, fontweight='bold')
        plt.title('四元组随时间变化：平均得分统计', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left')
        plt.xlabel('时间', fontsize=12, fontweight='bold')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        
        score_output_path = os.path.join(output_dir, f'{filename_base}_score_over_time.png')
        plt.savefig(score_output_path, dpi=300)
        print(f"平均得分图表已保存至: {score_output_path}")
        plt.close()

if __name__ == "__main__":
    main()
