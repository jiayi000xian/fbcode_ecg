#!/usr/bin/env python3

from fblearner.flow.api import types

EVAL_ARGS_SCHEME = types.STRUCT(
    ("data_path", types.TEXT),
    ("test_file", types.TEXT),
    ("train_file", types.TEXT),
    ("valid_file", types.TEXT),
    ("label_order", types.TEXT),
    ("read_per_line", types.BOOL),
    ("model_name_or_path", types.TEXT),
    ("from_checkpoint", types.TEXT),
    ("model_save_name", types.TEXT),
    ("checkpoint_upload_interval", types.INTEGER),
    ("output_upload_interval", types.INTEGER),
    ("device", types.TEXT),
    ("max_i_length", types.INTEGER),
    ("max_o_length", types.INTEGER),
    ("replace_underscores", types.BOOL),
    ("data_parallel", types.BOOL),
    ("use_proxy", types.BOOL),
    ("train_batch_size", types.INTEGER),
    ("eval_batch_size", types.INTEGER),
    ("eval_every_k_epoch", types.INTEGER),
    ("backward_freq", types.INTEGER),
    ("print_freq", types.INTEGER),
    ("learning_rate", types.FLOAT),
    ("num_epochs", types.INTEGER),
    ("decode_beams", types.INTEGER),
    ("use_multisoftmax", types.BOOL),
    ("checkpoint_save_option", types.TEXT),
    ("decode_on_lattice", types.BOOL),
    ("num_thresholds", types.INTEGER),
    ("min_support", types.FLOAT),
    ("output_dir", types.TEXT),
    ("get_curve", types.TEXT),
    ("show_examples", types.BOOL),
)

import argparse


class ArgumentsS2S(argparse.ArgumentParser):
    def __init__(
        self,
        add_s2s_args=True,
        add_s2s_dataset_args=True,
        description="S2S parser",
    ):
        super().__init__(
            description=description,
            allow_abbrev=False,
            conflict_handler="resolve",
            formatter_class=argparse.HelpFormatter,
            add_help=add_s2s_args,
        )

        if add_s2s_args:
            self.add_s2s_args()
        if add_s2s_dataset_args:
            self.add_s2s_dataset_args()

    def add_s2s_dataset_args(self, args=None):
        parser = self.add_argument_group("Dataset Arguments")
        parser.add_argument(
            "--data_path",
            default="manifold://fast_content_understanding/tree/ecg/data-bin/debug",
            type=str,
        )
        parser.add_argument(
            "--train_file", default="train", type=str, help="input the train file name"
        )
        parser.add_argument(
            "--test_file", default="test", type=str, help="input the test file name"
        )
        parser.add_argument(
            "--valid_file", default="valid", type=str, help="input the valid file name"
        )
        parser.add_argument("--label_order", default="random", type=str)
        parser.add_argument(
            "--read_per_line",
            default=True,
            type=bool,
            help="True for jsonl file, False for json file",
        )

    def add_s2s_args(self, args=None):
        parser = self.add_argument_group("Common Arguments")
        parser.add_argument(
            "--model_name_or_path",
            default="manifold://fast_content_understanding/tree/ecg/model/t5_small",
            type=str,
            help="Pretrained model name or path",
        )
        parser.add_argument(
            "--model_save_name",
            type=str,
            default="test.cp",
            help="Model output path",
        )
        parser.add_argument(
            "--device",
            default="cuda",
            type=str,
            help="Device: CPU or CUDA",
        )
        parser.add_argument(
            "--max_i_length",
            default=256,
            type=int,
            help="Max input length",
        )
        parser.add_argument(
            "--max_o_length",
            default=128,
            type=int,
            help="Max output length",
        )
        parser.add_argument(
            "--replace_underscores",
            default=True,
            type=bool,
        )
        parser.add_argument(
            "--data_parallel",
            action="store_true",
        )
        parser.add_argument(
            "--use_proxy",
            action="store_true",
        )

    def add_train_and_eval_args(self, args=None):
        parser = self.add_argument_group("Model training and evaluation arguments")
        parser.add_argument(
            "--train_batch_size",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--backward_freq",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--print_freq",
            default=20,
            type=int,
        )
        parser.add_argument(
            "--learning_rate",
            default=2e-4,
            type=float,
        )
        parser.add_argument(
            "--num_epochs",
            default=200,
            type=int,
        )
        parser.add_argument(
            "--eval_batch_size",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--decode_beams",
            default=3,
        )
        parser.add_argument(
            "--eval_every_k_epoch",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--use_multisoftmax",
            default=True,
            type=bool,
        )
        parser.add_argument(
            "--checkpoint_save_option",
            default="best",
            type=str,
        )
        parser.add_argument(
            "--from_checkpoint",
            default="manifold://jiayi000xian_bucket/tree/checkpoint/t5b_eur_lex/ft_rand_nloss_t5b_EUR_epoch_29_dir",
            type=str,
        )
        parser.add_argument(
            "--checkpoint_upload_interval",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--output_upload_interval",
            default=1,
            type=int,
        )

        parser.add_argument(
            "--decode_on_lattice",
            default=False,
            type=bool,
        )

        parser.add_argument(
            "--output_dir",
            default="evaluation",
            type=str,
        )

        parser.add_argument(
            "--num_thresholds",
            default=3,
            type=int,
        )

        parser.add_argument(
            "--min_support",
            default=0,
            type=float,
        )

        parser.add_argument(
            "--show_examples",
            default=True,
            type=bool,
        )

        parser.add_argument(
            "--get_curve",
            default="curve_only",
            type=str,
        )
