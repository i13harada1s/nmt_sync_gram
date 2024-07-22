import math
from typing import Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)


def hinge_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
    mask: Optional[torch.Tensor] = None,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """Hinge pairwise loss function."""
    losses = F.relu(margin - (targets.sign() * logits))

    if mask is not None:
        losses = losses.masked_fill(mask, 0.0)

    if reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.mean()
    else:
        return losses


def rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 1.0,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """Pairwise learning-to-rank loss (shen et. al., 2018)
    see paper: https://arxiv.org/pdf/1806.04168.pdf,
        code: https://github.com/hantek/distance-parser
    """
    # compute span score: (B x S x 1) -> (B x S x S)
    logits = logits - logits.transpose(-2, -1)
    targets = targets - targets.transpose(-2, -1)

    mask = torch.ones_like(targets).triu(0).bool()
    
    losses = hinge_loss(
        logits=logits,
        targets=targets,
        margin=margin,
        mask=mask,
        reduction=None
    )
    
    if reduction == "sum":
        return losses.sum()
    elif reduction == "mean":
        return losses.sum() / (mask==False).sum()
    else:
        return losses


@register_criterion("label_smoothed_cross_entropy_with_sync")
class LabelSmoothedCrossEntropyCriterionWithSync(LabelSmoothedCrossEntropyCriterion):
    """ """

    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        sync_lambda,
        sync_method,
        hinge_margen,
    ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.sync_lambda = sync_lambda
        self.sync_method = sync_method
        self.hinge_margen = hinge_margen

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--sync-lambda",
            default=0.05,
            type=float,
            metavar="D",
            help="weight to control the synchrononized loss",
        )
        parser.add_argument(
            "--sync-method",
            default="mse",
            type=str,
            metavar="STR",
            help="method to synchronize encoder outputs to decoder ones",
        )
        parser.add_argument(
            "--hinge-margen",
            default=1.0,
            type=float,
            metavar="D",
            help="margen for the hinge loss",
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "nll_loss": utils.item(nll_loss.data) if reduce else nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # Compute synchronized loss only for training set and non dummy batches.
        sync_loss = self.compute_sync_loss(model, net_output, sample)

        if sync_loss is not None:
            logging_output["sync_loss"] = utils.item(sync_loss.data)
            loss += self.sync_lambda * sync_loss

        return loss, sample_size, logging_output

    def compute_sync_loss(self, model, net_output, sample):
        loss = None

        if not hasattr(model, "get_sync_outputs"):
            return loss

        enc_out, dec_out = model.get_sync_outputs(net_output)

        padding_mask = sample["target"].eq(self.padding_idx).unsqueeze(-1)

        if enc_out.shape == dec_out.shape:
            if self.sync_method == "mse":
                loss = F.mse_loss(
                    enc_out.masked_fill(padding_mask, 0.0),
                    dec_out.masked_fill(padding_mask, 0.0),
                    reduction="sum",
                )
            else:  # family of hinge loss
                loss = rank_loss(
                    enc_out.masked_fill(padding_mask, 0.0),
                    dec_out.masked_fill(padding_mask, 0.0),
                    margin=self.hinge_margen,
                    reduction="sum",
                )
        return loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss_sum = utils.item(
            sum(log.get("nll_loss", 0) for log in logging_outputs)
        )
        sync_loss_sum = utils.item(
            sum(log.get("sync_loss", 0) for log in logging_outputs)
        )
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "sync_loss",
            sync_loss_sum / sample_size / math.log(2),
            sample_size,
            round=3,
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
