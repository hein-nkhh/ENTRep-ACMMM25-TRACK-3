import torch
import torch.nn as nn
import torch.nn.functional as F
# from utils.logger import default_logger as logger

class ContrastiveLoss(nn.Module):
    """ 
    Symmetric cross-entropy contrastive loss with rank-based penalty.
    Maximizes cosine similarity for matched image-text pairs,
    and penalizes if top-k ranking is violated.
    """
    def __init__(self, temperature=0.07, penalty_weight = 1.0):
        super().__init__()
        self.temperature = temperature
        self.penalty_weight = penalty_weight
        # logger.info("📐 ContrastiveLoss initialized with temperature=%.2f, penalty_weight=%.2f",
                    # temperature, penalty_weight)
        
    def forward(self, image_embedding, text_embedding, topk=5):
        batch_size = image_embedding.shape[0]
        labels = torch.arange(batch_size, device=image_embedding.device)

        # Cosine similarity scaled by temperature
        logits = torch.matmul(image_embedding, text_embedding.T) / self.temperature

        # Standard contrastive loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        total_loss = (loss_i2t + loss_t2i) / 2
        
        # Rank-based penalty: i2t
        ranks_i2t = (logits > logits[range(batch_size), labels].unsqueeze(1)).sum(dim=1)
        penalty_i2t = (ranks_i2t >= topk).float().mean()

        # Rank-based penalty: t2i
        ranks_t2i = (logits.T > logits.T[range(batch_size), labels].unsqueeze(1)).sum(dim=1)
        penalty_t2i = (ranks_t2i >= topk).float().mean()

        rank_penalty = (penalty_i2t + penalty_t2i) / 2
        total_loss += rank_penalty * self.penalty_weight

        # Accuracy for t2i (optional metric)
        acc_t2i = (torch.argmax(logits, dim=0) == labels).float().mean()
        
        return total_loss, acc_t2i