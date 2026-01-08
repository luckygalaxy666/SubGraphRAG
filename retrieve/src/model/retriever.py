import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from torch_geometric.nn import MessagePassing

class TemporalPEConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, edge_index, x, edge_weights):
        row, col = edge_index
        deg = torch_scatter.scatter_add(edge_weights, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        norm = deg_inv[row] * edge_weights
        
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class DDE(nn.Module):
    def __init__(
        self,
        num_rounds,
        num_reverse_rounds,
        alpha=0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for _ in range(num_rounds):
            self.layers.append(TemporalPEConv())
        
        self.reverse_layers = nn.ModuleList()
        for _ in range(num_reverse_rounds):
            self.reverse_layers.append(TemporalPEConv())
        
        self.alpha = alpha
    
    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        edge_ts,
        avg_ts_per_node
    ):
        result_list = []
        h_id_tensor, t_id_tensor = edge_index[0], edge_index[1]

        # Calculate weights for the forward pass based on the source nodes' average timestamp.
        avg_ts_for_source_nodes = avg_ts_per_node[h_id_tensor]
        time_diff_fwd = torch.abs(edge_ts - avg_ts_for_source_nodes)
        edge_weights_fwd = torch.exp(-self.alpha * time_diff_fwd)

        # Calculate weights for the reverse pass based on the target nodes' (which are the sources in reverse) average timestamp.
        avg_ts_for_target_nodes = avg_ts_per_node[t_id_tensor]
        time_diff_rev = torch.abs(edge_ts - avg_ts_for_target_nodes)
        edge_weights_rev = torch.exp(-self.alpha * time_diff_rev)
        
        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe, edge_weights_fwd)
            result_list.append(h_pe)
        
        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            h_pe_rev = layer(reverse_edge_index, h_pe_rev, edge_weights_rev)
            result_list.append(h_pe_rev)
        
        return result_list

class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        temporal_alpha=0.1
    ):
        super().__init__()
        
        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        DDE_kwargs['alpha'] = temporal_alpha
        self.dde = DDE(**DDE_kwargs)
        self.time_encoder = nn.Linear(1, emb_size)
        
        pred_in_size = 5 * emb_size # h_q, h_e[h], h_r, h_e[t], h_ts
        if topic_pe:
            pred_in_size += 2 * 2
        pred_in_size += 2 * 2 * (DDE_kwargs['num_rounds'] + DDE_kwargs['num_reverse_rounds'])

        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1)
        )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        ts_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot
    ):
        device = entity_embs.device
        if entity_embs.shape[-1] != self.non_text_entity_emb.weight.shape[-1]:
            entity_embs = entity_embs[:, :self.non_text_entity_emb.weight.shape[-1]]
        h_e = torch.cat(
            [
                entity_embs,
                self.non_text_entity_emb(
                    torch.LongTensor([0]).to(device)).expand(num_non_text_entities, -1)
            ]
        , dim=0)
        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([
            h_id_tensor,
            t_id_tensor
        ], dim=0)
        reverse_edge_index = torch.stack([
            t_id_tensor,
            h_id_tensor
        ], dim=0)
        
        num_entities = h_e.shape[0]

        # ===== Start of Centralized Calculation =====
        # For each node, calculate the average timestamp of all its connected edges ONCE.
        source_nodes, dest_nodes = edge_index[0], edge_index[1]
        all_nodes = torch.cat([source_nodes, dest_nodes])
        all_ts_repeated = torch.cat([ts_id_tensor, ts_id_tensor])
        avg_ts_per_node = torch_scatter.scatter_mean(all_ts_repeated, all_nodes, dim=0, dim_size=num_entities)
        # ===== End of Centralized Calculation =====

        # Pass the pre-computed avg_ts_per_node to DDE
        dde_list_raw = self.dde(topic_entity_one_hot, edge_index, reverse_edge_index, ts_id_tensor, avg_ts_per_node)
        # Normalize each tensor in the dde list to prevent exploding features
        dde_list_normalized = [F.normalize(t, p=2, dim=1) for t in dde_list_raw]
        h_e_list.extend(dde_list_normalized)
        
        h_e = torch.cat(h_e_list, dim=1)

        h_q = q_emb
        # Potentially memory-wise problematic
        h_r = relation_embs[r_id_tensor]
        
        # Encode the timestamp for each triple, now considering temporal context
        # We reuse the avg_ts_per_node calculated above.
        
        # For each triple, get the avg timestamps of its head and tail entities.
        head_avg_ts = avg_ts_per_node[h_id_tensor]
        tail_avg_ts = avg_ts_per_node[t_id_tensor]
        
        # Calculate an aggregated time difference.
        ts_id_tensor_float = ts_id_tensor.float()
        head_time_diff = torch.abs(ts_id_tensor_float - head_avg_ts)
        tail_time_diff = torch.abs(ts_id_tensor_float - tail_avg_ts)
        avg_time_diff = (head_time_diff + tail_time_diff) / 2.0
        
        # Create a temporal context weight based on the difference.
        temporal_context_weight = torch.exp(-self.dde.alpha * avg_time_diff)

        # Encode the triple's own timestamp and apply the context weight.
        base_h_ts = self.time_encoder(ts_id_tensor.unsqueeze(-1))
        h_ts = base_h_ts * temporal_context_weight.unsqueeze(-1)


        h_triple = torch.cat([
            h_q.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor],
            h_ts
        ], dim=1)
        
        pred_triple_logits = self.pred(h_triple)
        return pred_triple_logits
