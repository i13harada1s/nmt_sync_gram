import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
)
from src.modules.structformer_layer import (
    GrammarInductionHead,
    StructformerDecoderLayer,
    StructformerEncoderLayer,
)
from torch import Tensor


@register_model("structformer")
class StructformerModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.alignment_heads = args.alignment_heads
        self.alignment_layer = args.alignment_layer

        self.supports_grammar_induction = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--inductor-activation-fn',
            choices=utils.get_available_activation_fns(),
            help='activation function to use',
        )
        parser.add_argument(
            '--inductor-dropout', 
            type=float, 
            metavar='D',
            help='dropout probability',
        )
        parser.add_argument(
            '--inductor-attention-dropout', 
            type=float, 
            metavar='D',
            help='dropout probability for attention weights',
        )
        parser.add_argument(
            '--inductor-activation-dropout', 
            type=float, 
            metavar='D',
            help='dropout probability after activation in FFN.'
        )
        parser.add_argument(
            '--inductor-ffn-embed-dim', 
            type=int, 
            metavar='N',
            help='encoder embedding dimension for FFN',
        )
        parser.add_argument(
            '--inductor-layers', 
            type=int, 
            metavar='N',
            help='num encoder layers',
        )
        parser.add_argument(
            '--inductor-attention-heads', 
            type=int, metavar='N',
            help='num encoder attention heads',
        )
        parser.add_argument(
            '--inductor-normalize-before', 
            action='store_true',
            help='apply layernorm before each encoder block',
        )
        parser.add_argument(
            "--inductor-head-attention-heads", 
            type=int, 
            metavar="N",
            help="num gated self-attention heads to induce the grammar.",
        )
        parser.add_argument(
            "--inductor-head-temperature", 
            type=float, 
            metavar="D",
            help="sensitivity of gap between adjacented syntactic distance.",
        )
        parser.add_argument(
            "--inductor-head-activation-fn", 
            choices=utils.get_available_activation_fns(),
            help="name of activation function to use for grammar indcution.",
        )
        parser.add_argument(
            '--alignment-heads', 
            type=int, 
            metavar='D',
            help='number of cross attention heads per layer to supervised with alignments.')
        parser.add_argument(
            '--alignment-layer', 
            type=int, 
            metavar='D',
            help='number of layers which has to be supervised. 0 corresponding to the bottommost layer.'
        )
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        structformer(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return StructformerEncoder(
            args,
            src_dict,
            embed_tokens,
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return StructformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def get_sync_outputs(
        self,
        net_output: Tuple[
            torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]
        ],
    ):
        enc_out = net_output[1]["encoder_distance"][0]
        dec_out = net_output[1]["decoder_distance"][0]
        enc_dec_attn = net_output[1]["attn"][0]

        # (B x D x E) x (B x E x 1) -> (B x D x 1)
        prj_out = torch.bmm(enc_dec_attn, enc_out)

        return prj_out, dec_out


class StructformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        # ADDED: override args for inductor
        override_args = argparse.Namespace(**vars(args))
        override_args.activation_fn = args.inductor_activation_fn
        override_args.dropout = args.inductor_dropout
        override_args.attention_dropout = args.inductor_attention_dropout
        override_args.activation_dropout = args.inductor_activation_dropout
        override_args.encoder_attention_heads = args.inductor_attention_heads
        override_args.encoder_normalize_before = args.inductor_normalize_before

        # ADDED: initialization for induction layer
        self.induction_layers = nn.ModuleList([])
        self.induction_layers.extend(
            [
                self.build_encoder_layer(override_args)
                for _ in range(args.inductor_layers)
            ]
        )
        del override_args

        # ADDED: initialization for head of inductor
        self.induction_head = GrammarInductionHead(
            input_dim=args.encoder_embed_dim,
            num_attn_heads=args.encoder_attention_heads,
            num_gate_heads=args.inductor_head_attention_heads,
            tau=args.inductor_head_temperature,
            activation_fn=args.inductor_head_activation_fn,
        )

    def build_encoder_layer(self, args):
        return StructformerEncoderLayer(args)

    def residual_connection(self, x, residual, scale=1.0):
        return (residual + x) * scale

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x1 = x.transpose(0, 1)  # inputs for extractor
        x2 = x.transpose(0, 1)  # inputs for encoder

        encoder_states = []

        # extracter layers
        for idx in range(len(self.induction_layers)):
            x1 = self.induction_layers[idx](
                x1, encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
        attn_gate, encoder_distance = self.induction_head(x1)

        # encoder layers
        if return_all_hiddens:
            encoder_states.append(x2)
        for idx, layer in enumerate(self.layers):
            x2 = layer(
                x2,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                attn_gate=attn_gate if idx == 0 else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x2)

        if self.layer_norm is not None:
            x2 = self.layer_norm(x2)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x2],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_distance": [encoder_distance],  # B x T x 1
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_distance"]) == 0:
            new_encoder_distance = []
        else:
            new_encoder_distance = [
                encoder_out["encoder_distance"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_distance": new_encoder_distance,  # B x T x 1
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }


class StructformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        override_args = argparse.Namespace(**vars(args))
        override_args.activation_fn = args.inductor_activation_fn
        override_args.dropout = args.inductor_dropout
        override_args.attention_dropout = args.inductor_attention_dropout
        override_args.activation_dropout = args.inductor_activation_dropout
        override_args.decoder_attention_heads = args.inductor_attention_heads
        override_args.decoder_normalize_before = args.inductor_normalize_before

        # ADDED: initialization for induction layer
        self.induction_layers = nn.ModuleList([])
        self.induction_layers.extend(
            [
                self.build_decoder_layer(override_args, no_encoder_attn=True)
                for _ in range(args.inductor_layers)
            ]
        )
        del override_args

        # ADDED: initialization for induction head layer
        self.induction_head = GrammarInductionHead(
            input_dim=args.decoder_embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            num_gate_heads=args.inductor_head_attention_heads,
            tau=args.inductor_head_temperature,
            activation_fn=args.inductor_head_activation_fn,
            mask_triu=True,
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return StructformerDecoderLayer(args, no_encoder_attn)

    def residual_connection(self, x, residual, scale=1.0):
        return (residual + x) * scale

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            # follows "Jointly Learning to Align and Translate with Transformer Models" (Garg et al., 2019)
            alignment_layer = self.num_layers - 2

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]
        encoder_distance = None
        if encoder_out is not None and len(encoder_out["encoder_distance"]) > 0:
            encoder_distance = encoder_out["encoder_distance"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # B x T x C -> T x B x C
        x1 = x.transpose(0, 1)
        x2 = x.transpose(0, 1)

        # extracter layers
        attn_gate = []
        decoder_distance = []
        for idx in range(len(self.induction_layers)):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x1)
            else:
                self_attn_mask = None
            x1, _, _ = self.induction_layers[idx](
                x1,
                encoder_out=None,
                encoder_padding_mask=padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )
        attn_gate, decoder_distance = self.induction_head(x1, incremental_state)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x2]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x2)
            else:
                self_attn_mask = None

            x2, layer_attn, _ = layer(
                x2,
                encoder_out=enc,
                encoder_padding_mask=padding_mask,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_gate=attn_gate if idx == 0 else None,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x2)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x2 = self.layer_norm(x2)

        # T x B x C -> B x T x C
        x2 = x2.transpose(0, 1)

        if self.project_out_dim is not None:
            x2 = self.project_out_dim(x2)

        outputs = {
            "attn": [attn],
            "inner_states": inner_states,
            "encoder_distance": [encoder_distance],
            "decoder_distance": [decoder_distance],
        }

        return x2, outputs

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state is not None:
            x = x.transpose(0, 1)
        return x


@register_model_architecture("structformer", "structformer")
def structformer(args):
    args.inductor_activation_fn = getattr(args, "inductor_activation_fn", "relu")
    args.inductor_dropout = getattr(args, "inductor_dropout", 0.0)
    args.inductor_attention_dropout = getattr(args, "inductor_attention_dropout", 0.0)
    args.inductor_activation_dropout = getattr(args, "inductor_activation_dropout", 0.0)
    args.inductor_layers = getattr(args, "inductor_layers", 1)
    args.inductor_attention_heads = getattr(args, "inductor_attention_heads", 8)
    args.inductor_normalize_before = getattr(args, "inductor_normalize_before", False)
    args.inductor_head_attention_heads = getattr(args, "inductor_head_attention_heads", 8)
    args.inductor_head_temperature = getattr(args, "inductor_head_temperature", 10.0)
    args.inductor_head_activation_fn = getattr(args, "inductor_head_activation_fn", "linear")
    args.alignment_heads = getattr(args, "alignment_heads", None)
    args.alignment_layer = getattr(args, "alignment_layer", None)
    base_architecture(args)

@register_model_architecture("structformer", "structformer_iwslt_de_en")
def structformer_iwslt_de_en(args):
    args.inductor_activation_fn = getattr(args, "inductor_activation_fn", "relu")
    args.inductor_dropout = getattr(args, "inductor_dropout", 0.0)
    args.inductor_attention_dropout = getattr(args, "inductor_attention_dropout", 0.0)
    args.inductor_activation_dropout = getattr(args, "inductor_activation_dropout", 0.0)
    args.inductor_layers = getattr(args, "inductor_layers", 1)
    args.inductor_attention_heads = getattr(args, "inductor_attention_heads", 4)
    args.inductor_normalize_before = getattr(args, "inductor_normalize_before", False)
    args.inductor_head_attention_heads = getattr(args, "inductor_head_attention_heads", 4)
    args.inductor_head_temperature = getattr(args, "inductor_head_temperature", 10.0)
    args.inductor_head_activation_fn = getattr(args, "inductor_head_activation_fn", "linear")
    args.alignment_heads = getattr(args, "alignment_heads", None)
    args.alignment_layer = getattr(args, "alignment_layer", None)
    transformer_iwslt_de_en(args)


@register_model_architecture("structformer", "structformer_vaswani_wmt_en_de_big")
def structformer_vaswani_wmt_en_de_big(args):
    args.inductor_activation_fn = getattr(args, "inductor_activation_fn", "relu")
    args.inductor_dropout = getattr(args, "inductor_dropout", 0.3)
    args.inductor_attention_dropout = getattr(args, "inductor_attention_dropout", 0.0)
    args.inductor_activation_dropout = getattr(args, "inductor_activation_dropout", 0.0)
    # args.inductor_ffn_embed_dim = getattr(args, "inductor_ffn_embed_dim", 2048)
    args.inductor_layers = getattr(args, "inductor_layers", 1)
    args.inductor_attention_heads = getattr(args, "inductor_attention_heads", 16)
    args.inductor_normalize_before = getattr(args, "inductor_normalize_before", False)
    args.inductor_head_attention_heads = getattr(args, "inductor_head_attention_heads", 16)
    args.inductor_head_temperature = getattr(args, "inductor_head_temperature", 10.0)
    args.inductor_head_activation_fn = getattr(args, "inductor_head_activation_fn", "linear")
    args.alignment_heads = getattr(args, "alignment_heads", None)
    args.alignment_layer = getattr(args, "alignment_layer", None)
    transformer_vaswani_wmt_en_de_big(args)


# https://aclanthology.org/2020.emnlp-main.42.pdf
@register_model_architecture("structformer", "structformer_align_chen2020")
def structformer_align_chen2020(args):
    args.inductor_activation_fn = getattr(args, "inductor_activation_fn", "relu")
    args.inductor_dropout = getattr(args, "inductor_dropout", 0.0)
    args.inductor_attention_dropout = getattr(args, "inductor_attention_dropout", 0.0)
    args.inductor_activation_dropout = getattr(args, "inductor_activation_dropout", 0.0)
    args.inductor_layers = getattr(args, "inductor_layers", 1)
    args.inductor_attention_heads = getattr(args, "inductor_attention_heads", 4)
    args.inductor_normalize_before = getattr(args, "inductor_normalize_before", False)
    args.inductor_head_attention_heads = getattr(args, "inductor_head_attention_heads", 4)
    args.inductor_head_temperature = getattr(args, "inductor_head_temperature", 10.0)
    args.inductor_head_activation_fn = getattr(args, "inductor_head_activation_fn", "linear")
    args.alignment_heads = getattr(args, "alignment_heads", 4)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    transformer_iwslt_de_en(args)
