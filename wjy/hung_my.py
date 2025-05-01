import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from einops import rearrange


@torch.no_grad()
def hugarian_match(outputs, targets, p=1):
    """
    reference to the detr github repo
    @Params outputs: The namedtuple of the model output, as least have those entries
        "pred_logits": Tensor of dim [batch_size, num_queries, keyword_embeddings] with the classification logits
    @Params targets: The namedtuple of the model target, as least have those entries
        "embeddings": Tensor of dim [(batch_size, num_targets), keywords_embeddings] concated keywords batch,
        for num_targets is variable in different batches
        "target_length": Tensor of dim [batch_size] , contains the length of num_targets in different batches
    @Params p: The p of the p-norm for calculate the distance between outputs and targets, default is 2
    @Returns:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
        For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    """

    pred_logits = outputs.pred_logits
    target_embeddings = targets.embeddings

    B, Q, D = pred_logits.shape

    assert D == target_embeddings.shape[-1], (
        f"pred_logits and target_embeddings should have the same embedding dimension, but got {D} and {target_embeddings.shape[-1]} respectively."
    )

    pred_logits = rearrange(pred_logits, "b q d -> (b q) d")

    C = torch.cdist(pred_logits, target_embeddings, p=p)
    # [(b q), bt]
    C = rearrange(C, "(b q) bt -> b q bt", b=B, q=Q).cpu()

    sizes = [lgth.item() for lgth in targets.target_length]
    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return [
        (
            torch.as_tensor(i, dtype=torch.int64),
            torch.as_tensor(j, dtype=torch.int64),
        )
        for i, j in indices
    ]


if __name__ == "__main__":
    pred_logits = torch.rand(2, 100, 1024).cuda()
    target_embeddings = torch.rand(30, 1024).cuda()
    target_length = torch.tensor([20, 10], dtype=torch.int64)
    targets = type("Targets", (object,), {})()
    targets.embeddings = target_embeddings
    targets.target_length = target_length
    outputs = type("Outputs", (object,), {})()
    outputs.pred_logits = pred_logits
    indices = hugarian_match(outputs, targets)
    print(indices)
