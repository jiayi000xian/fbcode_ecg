import json
import os
from collections import defaultdict

import mwparserfromhell

from .data_processing_params import load_data_processing_args


# read Readme for folder and data organization
def read_data(path, read_per_line=True):
    with open(path, "r") as f:
        if read_per_line:
            lines = f.readlines()
            data = [json.loads(line) for line in lines]
        else:
            data = json.load(f)
        print("Complete reading data from %s" % (path))
    return data


def load_labels(input_path):
    labels = []
    try:
        with open(input_path, "r") as ip_fp:
            for line in ip_fp:
                labels.append(line)
    except UnicodeDecodeError:
        with open(input_path, "r", encoding="ISO-8859-1") as ip_fp:
            for line in ip_fp:
                labels.append(line)

    return labels  # TODO Do we need to erase the format symbols


def clean_tag(ip_tag):
    op_tag = ip_tag.split("->")[1]
    op_tag = op_tag.strip()
    op_tag = op_tag.replace("_", " ")
    return op_tag.strip()


def parse_data(
    labels, ip_path, op_path, process_func, read_per_line=False, to_print=10
):
    cnt = 0
    input_data = read_data(ip_path, read_per_line)
    with open(op_path, "w") as op_fp:
        for ip_ex in input_data:
            if cnt % 100000 == 0:
                print(f"{cnt} lines loaded!")
            op_ex = defaultdict()
            # ip_ex = json.loads(line)

            process_func(ip_ex, op_ex, labels)

            if cnt < to_print:
                print("===============================================")
                print(f"UID: {op_ex['sample_id']}")
                print(f"Input: {op_ex['input'][:2000]}")
                print(f"Output: {op_ex['output']}")
            op_fp.write(json.dumps(op_ex) + "\n")
            cnt += 1


def wiki_process(ip_ex, op_ex, labels):
    op_ex["sample_id"] = ip_ex["uid"]
    op_ex["input"] = (
        ip_ex["title"].replace("_", " ").strip()
        + " "
        + mwparserfromhell.parse(ip_ex["content"]).strip_code().strip()
    )
    op_ex["output"] = [clean_tag(labels[tag_idx]) for tag_idx in ip_ex["target_ind"]]
    return ip_ex, op_ex


def amazon_process(ip_ex, op_ex, labels):
    op_ex["sample_id"] = ip_ex["uid"]
    op_ex["input"] = ip_ex["title"].strip() + " " + ip_ex["content"].strip()
    op_ex["output"] = [labels[tag_idx].strip() for tag_idx in ip_ex["target_ind"]]
    return ip_ex, op_ex


def matcha_process(ip_ex, op_ex, labels=None):
    op_ex["sample_id"] = ip_ex["data_example_id"]
    if ip_ex["text"] == [] or ip_ex["all_entities"] == [] or ip_ex["gt_scores"] == []:
        return

    input_str = " ".join(ip_ex["text"])
    # for different training models, we may need to handle the "<EOS>" to match the end-of-sentence symbol in the tokeinzer' dictionary
    # input_str.replace("<EOS>", "</s>") mbart uses </s>
    # input_str.replace("<EOS>", "")
    op_ex["input"] = input_str
    label_num = len(ip_ex["all_entities"])
    op_ex["output"] = [
        ip_ex["all_entities"][i] for i in range(label_num) if ip_ex["gt_scores"][i] > 0
    ]


def data_preprocess():

    data_args = load_data_processing_args()
    print(data_args)

    print(
        "**** %s ****",
    )
    if data_args.label != "":
        labels_path = os.path.join(
            data_args.DATA, data_args.dataset, data_args.label + data_args.label_format
        )
        labels = load_labels(labels_path)
    else:
        labels = None

    print("Processing training data!")

    ip_path = os.path.join(data_args.DATA, data_args.dataset, data_args.file_raw)
    file_name = data_args.split + "." + data_args.split_format
    op_path = os.path.join(data_args.DATA, data_args.dataset, file_name)

    if "wiki" in data_args.dataset.lower():
        sample_process = wiki_process
    elif "amazon" in data_args.dataset.lower():
        sample_process = amazon_process
    elif "matcha" in data_args.dataset.lower():
        sample_process = matcha_process
    else:
        raise NotImplementedError

    parse_data(labels, ip_path, op_path, sample_process, data_args.read_per_line)

    print(
        "Preprocessing data from %s has completed, data saved to %s"
        % (ip_path, op_path)
    )


if __name__ == "__main__":
    data_preprocess()
