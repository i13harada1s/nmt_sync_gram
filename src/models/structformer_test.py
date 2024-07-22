import argparse

import torch
import numpy
from absl.testing import absltest
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.fairseq_task import LegacyFairseqTask
from fairseq.models import FairseqEncoderDecoderModel

from src.models.structformer import StructformerModel

DEFAULT_SRC_VOCAB_SIZE = 50
DEFAULT_TGT_VOCAB_SIZE = 100


def get_dummy_dictionary(vocab_size):
    dummy_dict = Dictionary()
    # add dummy symbol to satisfy vocab size
    for id, _ in enumerate(range(vocab_size)):
        dummy_dict.add_symbol("{}".format(id), 1000)
    return dummy_dict


class DummyTask(LegacyFairseqTask):
    def __init__(self, args):
        super().__init__(args)
        self.src_dict = get_dummy_dictionary(DEFAULT_SRC_VOCAB_SIZE)
        self.tgt_dict = get_dummy_dictionary(DEFAULT_TGT_VOCAB_SIZE)
        if getattr(self.args, "ctc", False):
            self.src_dict.add_symbol("<ctc_blank>")
            self.tgt_dict.add_symbol("<ctc_blank>")
        
    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


def get_dummy_task_and_parser():
    """
    to build a fariseq model, we need some dummy parse and task. This function
    is used to create dummy task and parser to faciliate model/criterion test
    Note: we use FbSpeechRecognitionTask as the dummy task. You may want
    to use other task by providing another function
    """
    parser = argparse.ArgumentParser(
        description="dummy_seq2seq_task", argument_default=argparse.SUPPRESS
    )
    DummyTask.add_args(parser)
    args = parser.parse_args([])
    task = DummyTask.setup_task(args)
    return task, parser


def get_dummy_input(batch_size=5, src_len=10, tgt_len=15, src_dim=25, tgt_dim=100):
    dummy_input = {}

    feature = torch.randint(high=src_dim, size=(batch_size, src_len), dtype=torch.int64)
    src_lengths = torch.randint(low=1, high=src_len, size=(batch_size, ), dtype=torch.int64)
    prev_output_tokens = torch.randint(high=src_dim, size=(batch_size, tgt_len), dtype=torch.int64)

    prev_output_tokens = fairseq_data_utils.collate_tokens(
        prev_output_tokens,
        pad_idx=1,
        eos_idx=2,
        left_pad=False,
        move_eos_to_beginning=False,
    )

    src_lengths, sorted_order = src_lengths.sort(descending=True)
    dummy_input["src_tokens"] = feature.index_select(0, sorted_order)
    dummy_input["src_lengths"] = src_lengths
    dummy_input["prev_output_tokens"] = prev_output_tokens

    return dummy_input


class StructformerTest(absltest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.src_len = 5
        self.tgt_len = 7
        self.src_dim = DEFAULT_SRC_VOCAB_SIZE
        self.tgt_dim = DEFAULT_TGT_VOCAB_SIZE

        self.setUpModel(StructformerModel)
        self.setUpInput(
            get_dummy_input(
                batch_size=self.batch_size, 
                src_len=self.src_len, 
                tgt_len=self.tgt_len, 
                src_dim=self.src_dim, 
                tgt_dim=self.tgt_dim,
            )
        )

    def setUpModel(self, model_cls, extra_args_setters=None):
        self.assertTrue(
            issubclass(model_cls, FairseqEncoderDecoderModel),
            msg="This class is only used for testing FairseqEncoderDecoderModel.",
        )
        task, parser = get_dummy_task_and_parser()
        model_cls.add_args(parser)
        args = parser.parse_args([])
        if extra_args_setters is not None:
            for args_setter in extra_args_setters:
                args_setter(args)
        model = model_cls.build_model(args, task)
        self.model = model

    def setUpInput(self, input=None):
        self.dummy_input = get_dummy_input() if input is None else input

    def test_model_forward(self):
        if self.model and self.dummy_input:
            output = self.model(**self.dummy_input)
            self.assertEqual(
                list(output[0].shape), [self.batch_size, self.tgt_len, self.tgt_dim + 4] # 4 = len([bos="<s>", pad="<pad>", eos="</s>", unk="<unk>"])
            )

    def test_model_get_sync_outputs(self):
        net_output = (
            torch.rand(self.batch_size, self.tgt_len, self.tgt_dim + 4),
            {
                "encoder_distance": [torch.rand(self.batch_size, self.src_len, 1)],
                "decoder_distance": [torch.rand(self.batch_size, self.tgt_len, 1)],
                "attn": [torch.rand(self.batch_size, self.tgt_len, self.src_len)],
            },
        )
        output = self.model.get_sync_outputs(net_output)

        self.assertEqual(list(output[0].shape), [self.batch_size, self.tgt_len, 1])
        self.assertEqual(list(output[1].shape), [self.batch_size, self.tgt_len, 1])

    # def test_encoder_forward(self):
    #     pass

    # def test_decoder_forward(self):
    #     pass


if __name__ == "__main__":
    absltest.main()
