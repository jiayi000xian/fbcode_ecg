import argparse


def load_data_processing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="matcha_f285499025/all",
        type=str,
        help="path to dataset",
    )

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="split: valid/train/test",
    )

    parser.add_argument(
        "--split_format",
        default="json",
        type=str,
        help="format of the split file(s)",
    )

    parser.add_argument(
        "--label",
        default="",
        type=str,
        help="common label file",
    )

    parser.add_argument(
        "--label_format",
        default="txt",
        type=str,
        help="format of the split file(s)",
    )

    parser.add_argument(
        "--file_raw",
        default="all_test.txt",
        type=str,
        help="format of the raw data file(s)",
    )

    parser.add_argument(
        "--DATA",
        default="/home/jiayi000xian/data-bin",
        type=str,
        help="path of data-bin",
    )

    parser.add_argument(
        "--read_per_line",
        default=False,
        type=bool,
        help="False for json file and True for jsonl",
    )

    args = parser.parse_args()
    return args
