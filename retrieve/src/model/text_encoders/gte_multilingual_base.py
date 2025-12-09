import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class GTEMultilingualBase:
    """多语言文本编码器，支持多语言实体和关系的嵌入生成"""
    def __init__(self,
                 device,
                 normalize=True,
                 model_path=None):
        self.device = device
        # 默认使用Alibaba-NLP的多语言嵌入模型，也可以使用其他多语言模型
        if model_path is None:
            model_path = 'Alibaba-NLP/gte-multilingual-base'
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            unpad_inputs=True,
            use_memory_efficient_attention=True).to(device)
        self.normalize = normalize

    @torch.no_grad()
    def embed(self, text_list):
        if len(text_list) == 0:
            return torch.zeros(0, 768)
        
        batch_dict = self.tokenizer(
            text_list, max_length=8192, padding=True,
            truncation=True, return_tensors='pt').to(self.device)
        
        outputs = self.model(**batch_dict).last_hidden_state
        emb = outputs[:, 0]
        
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=1)
        
        return emb.cpu()

    def __call__(self, q_text, text_entity_list, relation_list):
        q_emb = self.embed([q_text])
        entity_embs = self.embed(text_entity_list)
        relation_embs = self.embed(relation_list)
        
        return q_emb, entity_embs, relation_embs

