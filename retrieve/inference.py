import csv
import os
import re
import torch

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample

def sanitize_question_for_filename(question: str) -> str:
    # 去掉末尾问号（半角/全角），并替换文件名不安全字符
    question = question.rstrip('？?')
    safe = re.sub(r'[\\\\/<>:"\\|?*]', '_', question)
    safe = safe.strip()
    return safe or 'question'

def write_csv(pred_dict, ordered_ids, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            'question_id', 'question', 'rank', 'head', 'relation',
            'tail', 'time', 'score'
        ])

        for q_id in ordered_ids:
            sample = pred_dict[q_id]
            if not sample['scored_triples']:
                writer.writerow([q_id, sample['question'], '', '', '', '', '', ''])
                continue

            for rank, triple in enumerate(sample['scored_triples'], start=1):
                if len(triple) == 5:
                    head, relation, tail, time_val, score = triple
                else:
                    head, relation, tail, score = triple
                    time_val = ''
                writer.writerow([
                    q_id, sample['question'], rank, head, relation, tail, time_val,
                    f'{score:.6f}'
                ])

def write_split_csv(pred_dict, ordered_ids, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for q_id in ordered_ids:
        sample = pred_dict[q_id]
        base = sanitize_question_for_filename(sample['question'])
        csv_path = os.path.join(out_dir, f'{base}_sub.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['head', 'relation', 'tail', 'time', 'score'])
            if not sample['scored_triples']:
                writer.writerow(['', '', '', '', ''])
                continue
            for rank, triple in enumerate(sample['scored_triples'], start=1):
                if len(triple) == 5:
                    head, relation, tail, time_val, score = triple
                else:
                    head, relation, tail, score = triple
                    time_val = ''
                writer.writerow([head, relation, tail, time_val, f'{score:.6f}'])

@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0')
    
    cpt = torch.load(args.path, map_location='cpu')
    if not isinstance(cpt, dict):
        raise ValueError('未能识别的文件格式，请提供训练好的 checkpoint 或检索结果文件。')

    if 'config' not in cpt:
        # Allow exporting CSV directly from an existing retrieval_result.pth
        looks_like_result = all(
            isinstance(v, dict) and 'scored_triples' in v
            for v in cpt.values()
        )
        if looks_like_result:
            root_path = os.path.dirname(args.path)
            ordered_ids = list(cpt.keys())
            if args.split_export_csv:
                split_dir = os.path.join(root_path, 'tsmc')
                write_split_csv(cpt, ordered_ids, split_dir)
                return
            if args.export_csv:
                base_csv = os.path.splitext(os.path.basename(args.path))[0] + '.csv'
                write_csv(cpt, ordered_ids, os.path.join(root_path, base_csv))
                return
            raise KeyError('未找到 config。请传入训练好的 checkpoint 路径，或加上 --export_csv 并提供 retrieval_result.pth 以仅导出 CSV。')
        raise KeyError('未找到 config。请确认传入的是训练好的 checkpoint。')

    config = cpt['config']
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split=args.split, skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    ordered_ids = []
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        time_list = raw_sample.get('time_list', None)
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                h_txt = entity_list[h_id_tensor[triple_id].item()]
                r_txt = relation_list[r_id_tensor[triple_id].item()]
                t_txt = entity_list[t_id_tensor[triple_id].item()]
                if time_list is not None and triple_id < len(time_list):
                    time_val = time_list[triple_id]
                    top_K_triples.append((h_txt, r_txt, t_txt, time_val, top_K_scores[j]))
                else:
                    top_K_triples.append((h_txt, r_txt, t_txt, top_K_scores[j]))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }
        
        sample_id = raw_sample['id']
        pred_dict[sample_id] = sample_dict
        ordered_ids.append(sample_id)

    root_path = os.path.dirname(args.path)
    base_pth = 'retrieval_result.pth' if args.split == 'test' else f'retrieval_result_{args.split}.pth'
    result_path = os.path.join(root_path, base_pth)
    torch.save(pred_dict, result_path)

    if args.split_export_csv:
        split_dir = os.path.join(root_path, 'tsmc')
        write_split_csv(pred_dict, ordered_ids, split_dir)
    elif args.export_csv:
        base_csv = 'retrieval_result.csv' if args.split == 'test' else f'retrieval_result_{args.split}.csv'
        write_csv(pred_dict, ordered_ids, os.path.join(root_path, base_csv))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to run inference on (default: test)')
    parser.add_argument('--max_K', type=int, default=5000,
                        help='K in top-K triple retrieval')
    parser.add_argument('--export_csv', action='store_true',
                        help='Save the scored triples as a CSV file next to retrieval_result.pth')
    parser.add_argument('--split_export_csv', action='store_true',
                        help='Export each question to its own CSV under <checkpoint_dir>/tsmc')
    args = parser.parse_args()

    main(args)
