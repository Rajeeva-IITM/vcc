import torch
import torchmetrics

if __name__ == "__main__":
    a = torch.randn(3, 5)
    b = torch.randn_like(a)
    metric = torchmetrics.functional.mean_absolute_error(a, b)
    print(metric)
