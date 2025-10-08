import torch
import torch.nn.functional as F

from typing import List


def compute_joint_loss(rm_scores: torch.Tensor, correctness: torch.Tensor, candidates: List[List[str]]):
    """Joint optimization loss using pairwise reward model loss.

    Parameters
    ----------
    rm_scores : torch.Tensor
        Shape [B, N] reward model scores.
    correctness : torch.Tensor
        Shape [B, N] correctness labels (1/0) once implemented.
    candidates : List[List[str]]
        Generated candidate solution strings per question.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor for backpropagation.
    """
    batch_size = rm_scores.shape[0]

    # Convert correctness to tensor if it's a list
    if isinstance(correctness, list):
        # Convert list of lists to tensor
        max_candidates = max(len(corr) for corr in correctness)
        correctness_tensor = torch.zeros(batch_size, max_candidates, dtype=torch.float, device=rm_scores.device)
        for i, corr_list in enumerate(correctness):
            correctness_tensor[i, :len(corr_list)] = torch.tensor(corr_list, dtype=torch.float)
        correctness = correctness_tensor

    r_pos_list = []
    r_neg_list = []

    for b in range(batch_size):
        batch_scores = rm_scores[b]
        batch_correctness = correctness[b]

        # Find valid indices (non-zero scores indicate valid candidates)
        valid_mask = batch_scores != 0
        if not valid_mask.any():
            continue

        valid_scores = batch_scores[valid_mask]
        valid_correctness = batch_correctness[valid_mask]

        # Find correct answers (label = 1)
        correct_mask = valid_correctness == 1
        # Find incorrect answers (label = 0)
        incorrect_mask = valid_correctness == 0

        if correct_mask.any() and incorrect_mask.any():
            # Get correct answer with lowest RM score as r_pos (preserve gradients)
            correct_scores = valid_scores[correct_mask]
            min_idx = correct_scores.argmin()
            r_pos = correct_scores[min_idx]

            # Get incorrect answer with highest RM score as r_neg (preserve gradients)
            incorrect_scores = valid_scores[incorrect_mask]
            max_idx = incorrect_scores.argmax()
            r_neg = incorrect_scores[max_idx]

            r_pos_list.append(r_pos)
            r_neg_list.append(r_neg)

    if not r_pos_list:
        # No valid pairs found, return zero loss
        return torch.tensor(0.0, device=rm_scores.device, requires_grad=True)

    r_pos_tensor = torch.stack(r_pos_list)
    r_neg_tensor = torch.stack(r_neg_list)

    # Use pairwise RM loss
    loss = pairwise_rm_loss(r_pos_tensor, r_neg_tensor)

    return loss


def pairwise_rm_loss(
    r_pos: torch.Tensor,
    r_neg: torch.Tensor,
    *,
    margin: float = 0.0,
    temperature: float = 1.0,
    reduction: str = "mean",
):
    """
    Pairwise Bradley–Terry reward-model loss for (good vs bad) answers.

    Args:
        r_pos: Tensor of shape [B], RM scores for preferred/correct answers.
        r_neg: Tensor of shape [B], RM scores for dispreferred/incorrect answers.
        margin: Optional additive margin; encourages r_pos >= r_neg + margin.
        temperature: Scales (r_pos - r_neg - margin) / temperature.
        reduction: "mean" | "sum" | "none".
    Returns:
        loss: scalar if reduction != "none", else [B].
    """
    z = (r_pos - r_neg - margin) / temperature
    # -log σ(z)  ==  softplus(-z)
    loss_vec = F.softplus(-z)
    if reduction == "mean":
        return loss_vec.mean()
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec
