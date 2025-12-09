import argparse
import copy
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert retriever checkpoint to a new embedding size.')
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to original checkpoint (e.g., cwq_Dec01-12:00:37/cpt.pth)')
    parser.add_argument(
        '-o', '--output', required=True,
        help='Path to save converted checkpoint')
    parser.add_argument(
        '--target_emb', type=int, default=768,
        help='Target embedding size (default: 768)')
    return parser.parse_args()


def compute_struct_dim(retriever_cfg):
    topic_dim = 2 if retriever_cfg['topic_pe'] else 0
    dde_layers = (
        retriever_cfg['DDE_kwargs']['num_rounds'] +
        retriever_cfg['DDE_kwargs']['num_reverse_rounds']
    )
    dd_dim = 2 * dde_layers
    return topic_dim + dd_dim  # per-entity structural dims


def resize_non_text_emb(old_tensor, target_dim):
    return old_tensor[:, :target_dim].contiguous()


def resize_pred_layer(old_weight, old_bias, emb_old, emb_new, struct_dim):
    # Split column blocks: q | head | rel | tail
    idx = 0
    q_block = old_weight[:, idx: idx + emb_old]
    idx += emb_old

    head_block = old_weight[:, idx: idx + emb_old + struct_dim]
    idx += emb_old + struct_dim

    rel_block = old_weight[:, idx: idx + emb_old]
    idx += emb_old

    tail_block = old_weight[:, idx: idx + emb_old + struct_dim]

    head_emb = head_block[:, :emb_old]
    head_struct = head_block[:, emb_old:]
    tail_emb = tail_block[:, :emb_old]
    tail_struct = tail_block[:, emb_old:]

    q_new = q_block[:, :emb_new]
    head_emb_new = head_emb[:, :emb_new]
    rel_new = rel_block[:, :emb_new]
    tail_emb_new = tail_emb[:, :emb_new]

    new_weight_cols = torch.cat([
        q_new,
        torch.cat([head_emb_new, head_struct], dim=1),
        rel_new,
        torch.cat([tail_emb_new, tail_struct], dim=1)
    ], dim=1)

    new_weight = new_weight_cols[:emb_new, :].contiguous()
    new_bias = old_bias[:emb_new].contiguous()
    return new_weight, new_bias


def resize_pred_out(old_weight, emb_new):
    return old_weight[:, :emb_new].contiguous()


def main():
    args = parse_args()
    checkpoint = torch.load(args.input, map_location='cpu')
    config = checkpoint['config']
    retriever_cfg = config['retriever']

    state_dict = checkpoint['model_state_dict']
    emb_old = state_dict['non_text_entity_emb.weight'].shape[-1]
    target_emb = args.target_emb

    if emb_old == target_emb:
        print(f'Checkpoint already uses embedding size {target_emb}. '
              f'Copying file to {args.output}.')
        torch.save(checkpoint, args.output)
        return

    if emb_old < target_emb:
        raise ValueError(
            f'Cannot upcast from {emb_old} to {target_emb}. '
            'Please retrain the model instead.'
        )

    struct_dim = compute_struct_dim(retriever_cfg)

    new_state = copy.deepcopy(state_dict)

    # Resize non-text embeddings.
    new_state['non_text_entity_emb.weight'] = resize_non_text_emb(
        state_dict['non_text_entity_emb.weight'], target_emb)

    # Resize predictor layers.
    pred_in_weight = state_dict['pred.0.weight']
    pred_in_bias = state_dict['pred.0.bias']
    new_weight, new_bias = resize_pred_layer(
        pred_in_weight, pred_in_bias, emb_old, target_emb, struct_dim)
    new_state['pred.0.weight'] = new_weight
    new_state['pred.0.bias'] = new_bias

    new_state['pred.2.weight'] = resize_pred_out(
        state_dict['pred.2.weight'], target_emb)
    new_state['pred.2.bias'] = state_dict['pred.2.bias'].clone()

    checkpoint['model_state_dict'] = new_state
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f'Converted checkpoint saved to {args.output}')


if __name__ == '__main__':
    main()


