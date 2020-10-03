# encoding: utf-8


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)

    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
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
        >>> loss = DiceLoss()
        >>> input = torch.randn(3, 1, requires_grad=True)
        >>> target = torch.empty(3, dtype=torch.long).random_(5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-8,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 reduction: Optional[str] = "mean") -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
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

        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            return 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            return 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input,), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}"
