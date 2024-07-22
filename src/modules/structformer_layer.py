import math
from typing import Dict, List, Optional

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules import FairseqDropout, LayerNorm
from fairseq.modules.transformer_layer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from src import torch_utils
from src.modules.structformer_miltihead_attention import MultiheadAttention


@with_incremental_state
class GrammarInductionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        num_attn_heads,
        num_gate_heads,
        tau=1.0,
        activation_fn="linear",
        mask_triu=False,
    ):
        super().__init__()
        self.num_attn_heads = num_attn_heads
        self.num_gate_heads = num_gate_heads
        self.tau = tau
        self.mask_triu = mask_triu
        
        assert num_attn_heads >= num_gate_heads

        self.dense = Linear(input_dim, num_gate_heads)
        self.activation_fn = utils.get_activation_fn(activation_fn)

    def compute_span_logits(self, x):
        logits_float = torch_utils.sigmoid(((x - x.transpose(-2, -1)) * self.tau))

        span_logits_l_float = torch_utils.cumprod(
            logits_float.tril(-1) + torch.ones_like(logits_float).triu(), dim=-1, reverse=True
        )
        if self.mask_triu:
            return span_logits_l_float

        span_logits_r_float = torch_utils.cumprod(
            logits_float.triu(+1) + torch.ones_like(logits_float).tril(), dim=-1, reverse=False
        )
        return torch.mul(span_logits_l_float, span_logits_r_float)

    def forward(self, x, incremental_state=None):
        #  input: T x B x C
        # output: BH x T x T
        seq_size, batch_size = x.size(0), x.size(1)

        distance = self.activation_fn(self.dense(x))

        distance = (
            distance.contiguous()
            .view(-1, batch_size * self.num_gate_heads, 1)
            .transpose(0, 1)
        )  # BH x T x 1

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None:
                if "prev_distance" in saved_state:
                    prev_distance = saved_state["prev_distance"]
                    prev_distance = prev_distance.view(
                        batch_size * self.num_gate_heads, -1, 1
                    )
                    distance = torch.cat((prev_distance, distance), dim=1)
                saved_state["prev_distance"] = distance.view(
                    batch_size, self.num_gate_heads, -1, 1
                )
                incremental_state = self._set_input_buffer(
                    incremental_state, saved_state
                )

        attn_gates_float = self.compute_span_logits(distance)  # BH x T x T

        num_rmng_heads = self.num_attn_heads - self.num_gate_heads
        if num_rmng_heads > 0:
            attn_gates_float = torch.cat(
                [
                    attn_gates_float.view(
                        [batch_size, self.num_gate_heads, seq_size, seq_size]
                    ),
                    attn_gates_float.new_ones(
                        [batch_size, num_rmng_heads, seq_size, seq_size]
                    ),
                ],
                dim=1,
            ).view(
                [batch_size * self.num_attn_heads, seq_size, seq_size]
            )

        if incremental_state is not None:
            attn_gates_float = attn_gates_float[:, -1:, :]  # get the last element of sequence

        # average distance over heads
        distance = (
            distance.contiguous()
            .view(batch_size, self.num_gate_heads, -1, 1)
            .mean(dim=1)
        )  # B x T x 1

        return attn_gates_float, distance

    def reorder_incremental_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        new_order,
    ):
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.get_incremental_state(incremental_state, "head_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "head_state", buffer)


################################################################################


class StructformerEncoderLayer(TransformerEncoderLayer):
    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor] = None,
        attn_gate: Optional[torch.Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            attn_gate=attn_gate,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class StructformerDecoderLayer(TransformerDecoderLayer):
    def build_self_attention(
        self,
        embed_dim,
        args,
        add_bias_kv=False,
        add_zero_attn=False,
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_gate: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[torch.Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=need_attn or (not self.training and self.need_attn),
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
            attn_gate=self_attn_gate,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


################################################################################


class ConvEncoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0.0,
        activation_fn="relu",
    ):
        super().__init__()
        padding = kernel_size // 2 if kernel_size % 2 == 1 else 0
        self.conv = ConvTBC(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            padding=padding,
        )
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layer_norm = LayerNorm(out_channels)

    def residual_connection(self, x, residual, scaler=1.0):
        return (residual + x) * scaler

    def forward(self, x, encoder_padding_mask=None):
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.transpose(0, 1).unsqueeze(-1), 0)
        if self.conv.kernel_size[0] % 2 == 1:
            # padding is implicit in the conv
            x = self.conv(x)
        else:
            padding_l = (self.conv.kernel_size[0] - 1) // 2
            padding_r = (self.conv.kernel_size[0]) // 2
            # padding_l = self.conv.kernel_size[0] - 1
            # padding_r = 0
            x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
            x = self.conv(x)
        x = self.activation_fn(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.layer_norm(x)
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict


class ConvDecoderLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0.0,
        activation_fn="relu",
    ):
        super().__init__()
        self.conv = LinearizedConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            padding=kernel_size - 1,  # same as padding_l
        )
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layer_norm = LayerNorm(out_channels)

    def residual_connection(self, x, residual, scaler=1.0):
        return (residual + x) * scaler

    def forward(
        self,
        x,
        incremental_state: Optional[
            Dict[str, Dict[str, Optional[torch.Tensor]]]
        ] = None,
        **unused
    ):
        residual = x
        x = self.conv(x, incremental_state)
        x = self.activation_fn(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        x = self.layer_norm(x)
        return x


################################################################################


def Linear(in_features, out_features, dropout=0.0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m


def LinearizedConv1d(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer optimized for decoding"""
    from fairseq.modules import LinearizedConvolution

    m = LinearizedConvolution(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0.0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC

    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m
