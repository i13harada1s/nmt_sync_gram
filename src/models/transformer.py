from fairseq.models import register_model_architecture
from fairseq.models.transformer import (
    base_architecture,
    transformer_iwslt_de_en,
)


# https://arxiv.org/pdf/1909.02074.pdf -> cannot reproduce
@register_model_architecture("transformer_align", "transformer_align_garg2019")
def transformer_align_garg2019(args):
    args.alignment_heads = getattr(args, "alignment_heads", 8)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)
    
# https://aclanthology.org/2020.emnlp-main.42.pdf
@register_model_architecture("transformer_align", "transformer_align_chen2020")
def transformer_align_chen2020(args):
    args.alignment_heads = getattr(args, "alignment_heads", 4)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    transformer_iwslt_de_en(args)
    
# https://aclanthology.org/W19-5365.pdf
@register_model_architecture("transformer_align", "transformer_align_murakami2019")
def transformer_align_garg2019(args):
    args.alignment_heads = getattr(args, "alignment_heads", 8)
    args.alignment_layer = getattr(args, "alignment_layer", 4)
    args.full_context_alignment = getattr(args, "full_context_alignment", False)
    base_architecture(args)
