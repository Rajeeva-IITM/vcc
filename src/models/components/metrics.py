import numpy as np
import torch


def fold_change_metrics(
    y_true_log: torch.Tensor,
    y_pred_log: torch.Tensor,
    baseline_log: torch.Tensor,
    topk_cutoff: int,
):
    true_fc = y_true_log - baseline_log
    pred_fc = y_pred_log - baseline_log

    fc_mae = (true_fc - pred_fc).abs().mean()

    upregulated_genes_pred = torch.topk(pred_fc, topk_cutoff)[1]
    upregulated_genes_true = torch.topk(true_fc, topk_cutoff)[1]
    common = np.intersect1d(upregulated_genes_pred, upregulated_genes_true)

    downregulated_genes_pred = torch.topk(pred_fc, topk_cutoff)[1]
    downregulated_genes_true = torch.topk(true_fc, topk_cutoff)[1]

    return fc_mae, common, downregulated_genes_pred, downregulated_genes_true
