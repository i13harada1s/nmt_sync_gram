import torch
from fairseq import utils
from fairseq.sequence_scorer import SequenceScorer


class GrammarScorer(SequenceScorer):
    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]

        def batch_for_softmax(dec_out, target):
            # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
            first, rest = dec_out[0], dec_out[1:]
            batch_size, target_size, dim = first.shape
            if batch_size * target_size < self.softmax_batch:
                yield dec_out, target, True
            else:
                flat = first.contiguous().view(1, -1, dim)
                flat_tgt = target.contiguous().view(flat.shape[:-1])
                s = 0
                while s < flat.size(1):
                    e = s + self.softmax_batch
                    yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                    s = e

        def gather_target_probs(probs, target):
            probs = probs.gather(
                dim=2,
                index=target.unsqueeze(-1),
            )
            return probs

        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        avg_enc_dst = None
        avg_dec_dst = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            
            # FIXED: 
            outputs = decoder_out[1] if len(decoder_out) > 1 else None
            if type(outputs) is dict:
                attn = outputs.get("attn", None)
                enc_dst = outputs.get("encoder_distance", None)
                dec_dst = outputs.get("decoder_distance", None)
            else:
                raise RuntimeError("second return of model output must be dict object.")

            batched = batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                if is_single:
                    probs = gather_target_probs(curr_prob, orig_target)
                else:
                    if probs is None:
                        probs = curr_prob.new(orig_target.numel())
                    step = curr_prob.size(0) * curr_prob.size(1)
                    end = step + idx
                    tgt_probs = gather_target_probs(
                        curr_prob.view(tgt.shape + (curr_prob.size(-1),)), tgt
                    )
                    probs[idx:end] = tgt_probs.view(-1)
                    idx = end
                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
                
            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
                    
            # ADDED: process for encoder distance
            if enc_dst is not None:
                if torch.is_tensor(enc_dst):
                    enc_dst = enc_dst.data
                else:
                    enc_dst = enc_dst[0]
                if avg_enc_dst is None:
                    avg_enc_dst = enc_dst
                else:
                    avg_enc_dst.add_(enc_dst.data) 
                    
            # ADDED: process for decoder distance
            if dec_dst is not None:
                if torch.is_tensor(dec_dst):
                    dec_dst = dec_dst.data
                else:
                    dec_dst = dec_dst[0]
                if avg_dec_dst is None:
                    avg_dec_dst = dec_dst.data
                else:
                    avg_dec_dst.add_(dec_dst.data)
                    
        if len(models) > 1:
            avg_probs.div_(len(models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(models))
            # ADDED: avarage enc/dec distance
            if avg_enc_dst is not None:
                avg_enc_dst.div_(len(models))
            if avg_dec_dst is not None:
                avg_dec_dst.div_(len(models))

        batch_size = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * batch_size
        for i in range(batch_size):
            # remove padding from reference
            reference = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None else None
            )
            tgt_len = reference.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            # ADDED: extract valid enc/dec distance with torch.masked_select()
            if avg_enc_dst is not None:
                mask = (sample["net_input"]["src_tokens"][i] != self.pad).unsqueeze(-1)
                avg_enc_dst_i = torch.masked_select(avg_enc_dst[i], mask)[1:] # remove BOS
            else:
                avg_enc_dst_i = None
            if avg_dec_dst is not None:
                mask = (sample["target"][i] != self.pad).unsqueeze(-1)
                avg_dec_dst_i = torch.masked_select(avg_dec_dst[i], mask)[1:] # remove BOS
            else:
                avg_dec_dst_i = None
            hypos.append(
                [
                    {
                        "tokens": reference,
                        "score": score_i,
                        "encoder_span_score": avg_enc_dst_i,
                        "decoder_span_score": avg_dec_dst_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                    }
                ]
            )
        return hypos
