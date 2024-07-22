import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    DEFAULT_MIN_PARAMS_TO_WRAP,
    base_architecture,
    transformer_iwslt_de_en,
    transformer_vaswani_wmt_en_de_big,
)
from src.models.sdbnmt.model_layer import (
    GrammarInductionHead,
    CustomTransformerEncoderLayer,
    CustomTransformerDecoderLayer,
    ConvEncoderLayer,
    ConvDecoderLayer,
)
from src.models.sdbnmt.model_config import Seq2SeqModelConfig

@dataclass
class Seq2SeqModelConfig(TransformerModelConfig):
    encoder_conv_config: str = field(
        default="((512, 5), (512, 5))", 
        metadata={"help": ""}
    )
    decoder_conv_config: str = field(
        default="((512, 5), (512, 5))", 
        metadata={"help": ""}
    )
    encoder_conv_dropout: float = field(
        default=0.1, metadata={"help": ""}
    )
    decoder_conv_dropout: float = field(
        default=0.1, metadata={"help": ""}
    )
    encoder_head_config: str = field(
        default="((4, 0.1), (4, 0.1))", 
        metadata={"help": ""}
    )
    decoder_head_config: str = field(
        default="((4, 0.1), (4, 0.1))", 
        metadata={"help": ""}
    )
    head_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="linear", metadata={"help": "activation function name to use for head layer"}
    )
    
@register_model("structformer_conv", dataclass=Seq2SeqModelConfig)
class CustomTransformerModel(TransformerModel):
    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        self.alignment_heads = cfg.alignment_heads
        self.alignment_layer = cfg.alignment_layer
        
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        
        gen_parser_from_dataclass(parser, Seq2SeqModelConfig(), delete_default=True)
        
    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return CustomTransformerEncoder(
            cfg, 
            src_dict, 
            embed_tokens,
        )

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return CustomTransformerDecoder(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(cfg, "no_cross_attention", False),
        )
        
    def get_sync_outputs(
        self,
        net_output
    ):
        enc_out = net_output[1]["package"]["encoder_distance"]
        dec_out = net_output[1]["package"]["decoder_distance"]
        projector = net_output[1]["package"]["encoder_decoder_attention"]
        
        # enc_out = torch.softmax(enc_out, dim=-1)
        # dec_out = torch.softmax(dec_out, dim=-1)

        # (B x D x E) x (B x E x 1) -> (B x D x 1)
        prj_out = torch.einsum('bij,bje->bie', projector, enc_out)

        return prj_out, dec_out
    

class CustomTransformerEncoder(TransformerEncoder):
    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)
        
        # ADDED: initialization for feature extractor
        self.extra_layers = nn.ModuleList([])
        conv_config = eval(cfg.encoder_conv_config)
        input_dim = cfg.decoder_embed_dim
        for i in range(len(conv_config)):
            self.extra_layers.append(
                ConvEncoderLayer(
                    in_channels=input_dim,
                    out_channels=conv_config[i][0],
                    kernel_size=conv_config[i][1],
                    dropout=cfg.encoder_conv_dropout,
                ) 
            )
            input_dim = conv_config[i][0]
            
        # ADDED: initialization for induction head
        self.head_layers = nn.ModuleList([])
        head_config = eval(cfg.encoder_head_config)
        for i in range(len(head_config)):
            self.head_layers.append(
                GrammarInductionHead(
                    input_dim=cfg.encoder_embed_dim,
                    num_attn_heads=cfg.encoder_attention_heads,
                    num_gate_heads=head_config[i][0],
                    tau=head_config[i][1],
                    activation_fn=cfg.head_activation_fn,
                    mask_future=False,
                )
            )
        
    def build_encoder_layer(self, cfg):
        layer = CustomTransformerEncoderLayer(cfg)
        checkpoint = getattr(cfg, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(cfg, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(cfg, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def residual_connection(self, x, residual, scaler=1.0):
        return (residual + x) * scaler
        
    def forward_scriptable(
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
        x1 = x.transpose(0, 1) # inputs for parser

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x1)
            
        residual = x1
        
        # parser layers
        for idx, layer in enumerate(self.extra_layers):
            x1 = layer(
                x1, 
                encoder_padding_mask=encoder_padding_mask if has_pads else None
            )
        attn_gates, encoder_distance = [], []
        for idx, layer in enumerate(self.head_layers):
            attn_gate, distance = layer(x1)
            attn_gates.append(attn_gate)
            encoder_distance.append(distance)
        encoder_distance = torch.stack(encoder_distance, dim=0).sum(dim=0)
        
        x1 = self.residual_connection(x1, residual)

        # encoder layers
        for idx, layer in enumerate(self.layers):
            attn_gate = None
            if len(attn_gates) > idx:
                attn_gate = attn_gates[idx]
            x1 = layer(
                x1, 
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                attn_gate=attn_gate,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x1)

        if self.layer_norm is not None:
            x1 = self.layer_norm(x1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x1],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_distance": [encoder_distance], # B x T x 1
            "src_tokens": [],
            "src_lengths": [],
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
            "encoder_distance": new_encoder_distance, # B x T x 1
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    
class CustomTransformerDecoder(TransformerDecoder):
    def __init__(self, cfg, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn)

        self.extra_layers = nn.ModuleList([])
        conv_config = eval(cfg.decoder_conv_config)
        input_dim = cfg.decoder_embed_dim
        for i in range(len(conv_config)):
            self.extra_layers.append(
                ConvDecoderLayer(
                    in_channels=input_dim,
                    out_channels=conv_config[i][0],
                    kernel_size=conv_config[i][1],
                    dropout=cfg.decoder_conv_dropout,
                ) 
            )
            input_dim = conv_config[i][0]
        self.head_layers = nn.ModuleList([])
        head_config = eval(cfg.decoder_head_config)
        for i in range(len(head_config)):
            self.head_layers.append(
                GrammarInductionHead(
                    input_dim=cfg.decoder_embed_dim,
                    num_attn_heads=cfg.decoder_attention_heads,
                    num_gate_heads=head_config[i][0],
                    tau=head_config[i][1],
                    activation_fn=cfg.head_activation_fn,
                    mask_future=True,
                )
            )
        
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = CustomTransformerDecoderLayer(cfg, no_encoder_attn)
        checkpoint = getattr(cfg, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(cfg, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(cfg, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def build_extractor_layer(self, cfg, no_encoder_attn=False):
        layer = ExtraTransformerDecoderLayer(cfg, no_encoder_attn)
        checkpoint = getattr(cfg, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(cfg, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(cfg, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def residual_connection(self, x, residual, scaler=1.0):
        return (residual + x) * scaler
    
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
            # reference "Jointly Learning to Align and Translate with Transformer Models" (Garg et al., 2019)
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
        
        residual = x1
        
        # parser layers
        x1 = self._transpose_if_training(x, incremental_state)
        for idx, layer in enumerate(self.extra_layers):
            x1 = layer(x1, incremental_state)
            
        # head layers
        # x1 = self._transpose_if_inference(x1, incremental_state)
        attn_gates, decoder_distance = [], []
        for idx, layer in enumerate(self.head_layers):
            attn_gate, distance = layer(x1, incremental_state)
            attn_gates.append(attn_gate)
            decoder_distance.append(distance)
        decoder_distance = torch.stack(decoder_distance, dim=0).sum(dim=0)
        
        x1 = self.residual_connection(x1, residual)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x1]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x1)
            else:
                self_attn_mask = None
                
            self_attn_gate = None
            if len(attn_gates) > idx:
                self_attn_gate = attn_gates[idx]

            x1, layer_attn, _ = layer(
                x1,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_gate=self_attn_gate,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x1)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x1 = self.layer_norm(x)

        # T x B x C -> B x T x C
        x1 = x1.transpose(0, 1)

        if self.project_out_dim is not None:
            x1 = self.project_out_dim(x1)
          
        # tensor packages for computation of synchonous  
        package = {
            "encoder_distance": encoder_distance,
            "decoder_distance": decoder_distance,
            "encoder_decoder_attention": attn,
        }

        return x1, {"attn": [attn], "inner_states": inner_states, "package": package}
    
    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x
    
    def _transpose_if_inference(self, x, incremental_state):
        if incremental_state is not None:
            x = x.transpose(0, 1)
        return x

# @register_model_architecture("structformer", "structformer_iwslt_de_en")
# def structformer_iwslt_de_en(cfg):
#     cfg.encoder_gate_heads = getattr(cfg, "encoder_head_config", "((4, 0.1),) * 1")
#     cfg.decoder_gate_heads = getattr(cfg, "decoder_head_config", "((4, 0.1),) * 1")
#     cfg.head_activation_fn = getattr(cfg, "head_activation_fn", "linear")
    
#     cfg.encoder_conv_config = getattr(cfg, "encoder_conv_config", "((512, 9),) * 1")
#     cfg.decoder_conv_config = getattr(cfg, "decoder_conv_config", "((512, 9),) * 1")
#     cfg.encoder_conv_dropout = getattr(cfg, "encoder_conv_dropout", 0.3)
#     cfg.decoder_conv_dropout = getattr(cfg, "decoder_conv_dropout", 0.3)
    
#     transformer_iwslt_de_en(cfg)
    
# @register_model_architecture("structformer", "structformer_vaswani_wmt_en_de_big")
# def structformer_iwslt_de_en(cfg):
#     cfg.encoder_gate_heads = getattr(cfg, "encoder_head_config", "((4, 0.1),)")
#     cfg.decoder_gate_heads = getattr(cfg, "decoder_head_config", "((4, 0.1),)")
#     cfg.head_activation_fn = getattr(cfg, "head_activation_fn", "linear")
    
#     cfg.encoder_conv_config = getattr(cfg, "encoder_conv_config", "((1024, 9),)")
#     cfg.decoder_conv_config = getattr(cfg, "decoder_conv_config", "((1024, 9),)")
#     cfg.encoder_conv_dropout = getattr(cfg, "encoder_conv_dropout", 0.3)
#     cfg.decoder_conv_dropout = getattr(cfg, "decoder_conv_dropout", 0.3)
    
#     transformer_vaswani_wmt_en_de_big(cfg)