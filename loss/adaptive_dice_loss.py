# encoding: utf-8


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class AdaptiveDiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.

    Math Function:
        https://arxiv.org/abs/1911.02855.pdf
        adaptive_dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} (1 - p_i) ** alpha * p_i * y_i + smooth
            denominator = \sum_{1}^{t} (1 - p_i) ** alpha * p_i + \sum_{1} ^{t} y_i + smooth

    Args:
        alpha: alpha in math function
        smooth (float, optional): smooth in math function
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
            True: the loss combines a `sigmoid` layer and the `BCELoss` in one single class.
            False: the loss contains `BCELoss`.
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = AdaptiveDiceLoss()
        >>> input = torch.randn(3, 1, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 alpha: float = 0.1,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(AdaptiveDiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.alpha = alpha
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self,
                input: Tensor,
                target: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:

        flat_input = input.view(-1)
        flat_target = target.view(-1)

        if self.with_logits:
            flat_input = torch.sigmoid(flat_input)

        if mask is not None:
            mask = mask.view(-1).float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask

        intersection = torch.sum((1-flat_input)**self.alpha * flat_input * flat_target, -1) + self.smooth
        denominator = torch.sum((1-flat_input)**self.alpha * flat_input) + flat_target.sum() + self.smooth
        return 1 - 2 * intersection / denominator

    def __str__(self):
        return f"Adaptive Dice Loss, smooth:{self.smooth}; alpha:{self.alpha}"
