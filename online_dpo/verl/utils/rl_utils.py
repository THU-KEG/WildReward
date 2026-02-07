import torch
import torch.nn.functional as F

def get_log_probs(logits, labels, attention_mask):
    """
    计算序列的 log probability。
    logits: [B, Seq_Len, Vocab]
    labels: [B, Seq_Len] (通常就是 input_ids)
    attention_mask: [B, Seq_Len] (用于屏蔽 padding)
    """
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # [B, Seq_Len-1]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # Gather log probs of the actual labels
    # [B, Seq_Len-1]
    selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Apply mask (mask out padding)
    # shift mask accordingly
    shift_mask = attention_mask[..., 1:].contiguous()
    selected_log_probs = selected_log_probs * shift_mask
    
    return selected_log_probs

def compute_dpo_loss_batched(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0
):
    """
    DPO Loss 计算
    输入是 Sum 过的 LogProb (Scalar per sample)
    """
    pi_logratios = chosen_logps - rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios
    
    # DPO loss formula
    losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - \
             F.logsigmoid(-beta * logits) * label_smoothing
             
    # 用于监控的 rewards
    chosen_rewards = beta * (chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards, rejected_rewards