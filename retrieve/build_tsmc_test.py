import csv
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from src.model.text_encoders import GTEMultilingualBase


def _dedup_title(base: str) -> str:
    # 如果文件名包含逗号分隔的重复片段，压缩为一次
    parts = [p.strip() for p in base.replace('，', ',').split(',') if p.strip()]
    if not parts:
        return base.strip()
    if len(set(parts)) == 1:
        return parts[0]
    return '，'.join(parts)


def load_triples(csv_path: Path):
    triples = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            head = row[0]
            rel = row[2]
            tail = row[3] if len(row) > 3 else ''
            time_val = row[5] if len(row) > 5 else ''
            triples.append((head, rel, tail, time_val))
    return triples


def build_sample(csv_path: Path, default_entity: str = '台积电'):
    triples = load_triples(csv_path)
    # 确保问题实体/答案实体存在于节点列表
    all_entities = {h for h, _, _, _ in triples} | {t for _, _, t, _ in triples} | {default_entity}
    all_relations = {r for _, r, _, _ in triples}

    text_entity_list = sorted(all_entities)
    non_text_entity_list = []

    entity2id = {ent: idx for idx, ent in enumerate(text_entity_list)}
    relation_list = sorted(all_relations)
    rel2id = {rel: idx for idx, rel in enumerate(relation_list)}

    h_id_list, r_id_list, t_id_list, time_list = [], [], [], []
    for h, r, t, time_val in triples:
        h_id_list.append(entity2id[h])
        r_id_list.append(rel2id[r])
        t_id_list.append(entity2id[t])
        time_list.append(time_val)

    q_entities = [default_entity]
    a_entities = [default_entity]
    q_entity_id_list = [entity2id[default_entity]] if default_entity in entity2id else []
    a_entity_id_list = [entity2id[default_entity]] if default_entity in entity2id else []

    base_raw = csv_path.stem.replace('_processed', '')
    base = _dedup_title(base_raw)
    question = f'{base}？'
    sample_id = f'tsmc-{base}'

    return {
        'id': sample_id,
        'question': question,
        'q_entity': q_entities,
        'q_entity_id_list': q_entity_id_list,
        'text_entity_list': text_entity_list,
        'non_text_entity_list': non_text_entity_list,
        'relation_list': relation_list,
        'h_id_list': h_id_list,
        'r_id_list': r_id_list,
        't_id_list': t_id_list,
        'time_list': time_list,
        'a_entity': a_entities,
        'a_entity_id_list': a_entity_id_list
    }


def save_processed(samples, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(samples, f)
    print(f'Saved processed test set: {path} ({len(samples)} samples)')


def save_embeddings(samples, save_path: Path, device: str = 'cuda:0'):
    encoder = GTEMultilingualBase(torch.device(device))
    emb_dict = {}
    for sample in tqdm(samples, desc='Encoding'):
        q_emb, entity_embs, relation_embs = encoder(
            sample['question'],
            sample['text_entity_list'],
            sample['relation_list']
        )
        emb_dict[sample['id']] = {
            'q_emb': q_emb,
            'entity_embs': entity_embs,
            'relation_embs': relation_embs
        }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb_dict, save_path)
    print(f'Saved embeddings: {save_path}')


def main():
    data_dir = Path('data_files/tsmc')
    csv_files = sorted(data_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {data_dir}')

    samples = [build_sample(p) for p in csv_files]

    processed_path = data_dir / 'processed' / 'tsmc_test.pkl'
    save_processed(samples, processed_path)

    emb_path = data_dir / 'emb'/'gte-multilingual-base' / 'tsmc_test.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    save_embeddings(samples, emb_path, device=device)


if __name__ == '__main__':
    main()
