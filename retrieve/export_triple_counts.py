import argparse
import csv
import pickle
from pathlib import Path

from tqdm import tqdm


def count_triples(dataset: str, split: str, output: Path) -> None:
    data_path = Path('data_files') / dataset / 'processed' / f'{split}.pkl'
    if not data_path.exists():
        raise FileNotFoundError(f'未找到数据文件：{data_path}')

    with data_path.open('rb') as f:
        samples = pickle.load(f)

    rows = []
    for sample in tqdm(samples, desc='Counting triples'):
        # h_id_list/t_id_list/r_id_list 共同定义三元组数量；部分原始数据也包含 triple_list。
        triple_count = len(sample.get('h_id_list', sample.get('triple_list', [])))
        rows.append([
            sample.get('id', ''),
            triple_count,
            sample.get('question', '')
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id', 'triple_count', 'question'])
        writer.writerows(rows)

    print(f'Saved {len(rows)} rows to {output}')


def main():
    parser = argparse.ArgumentParser(
        description='统计指定数据集 split 中每个问题包含的三元组数量并写入 CSV。')
    parser.add_argument('-d', '--dataset', default='cwq',
                        help='数据集名称（默认为 cwq）')
    parser.add_argument('-s', '--split', default='train',
                        help='数据切分名，例如 train/val/test（默认为 train）')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 CSV 路径（默认与数据文件同目录下的 {split}_triple_counts.csv）')
    args = parser.parse_args()

    output = Path(args.output) if args.output else (
        Path('data_files') / args.dataset / 'processed' / f'{args.split}_triple_counts.csv')

    count_triples(args.dataset, args.split, output)


if __name__ == '__main__':
    main()
