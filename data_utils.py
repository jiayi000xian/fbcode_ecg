import copy
import json
import logging
import random
from collections import Counter
from itertools import chain

# import fblearner.flow.api as flow
# from aiplatform.modelstore.manifold import manifold_utils
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.fb.manifold import ManifoldPathHandler
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

PathManager = PathManagerBase()
PathManager.register_handler(
    ManifoldPathHandler(timeout_sec=1800, max_parallel=32), allow_override=True
)
RETENTION_DAYS = 90


class Seq2SetDataset(Dataset):
    def __init__(
        self,
        path,
        order,
        sep,
        random_seed=0,
        replace_underscores=False,
        read_per_line=False,
    ):
        self.path = path
        self.sep = sep
        self.data = None
        self.label_set = None
        self.replace_underscores = replace_underscores
        self.read_per_line = read_per_line
        self.label_count = None
        self.order = None
        self.order_init = order

        # dataset initialization
        self.read_data()
        self.get_label_count()
        self.order_rearrange_all(self.order_init)

    def read_data(self):

        with PathManager.open(self.path, "r") as f:
            if self.read_per_line:
                lines = f.readlines()
                self.data = [json.loads(line) for line in lines]
            else:
                self.data = json.load(f)

        print("Complete reading data from %s" % (self.path))

        self.label_set = set(chain.from_iterable(row["output"] for row in self.data))

    def get_label_count(self):
        if self.data is None:
            print("First read data from %s" % (self.path))
            self.read_data()

        self.label_count = Counter(
            chain.from_iterable(row["output"] for row in self.data)
        )

    def order_rearrange_all(self, order):

        if order == self.order:
            print(
                "Requset label order: %s is the same with the current label order: %s, no rearragment is made"
                % (order, self.order)
            )
            pass
        else:
            past_order = copy.deepcopy(self.order)
            self.order = order
            if order == "freq_increase":
                for sample in self.data:
                    sample["output"] = sorted(
                        sample["output"], key=lambda x: self.label_count[x]
                    )
            elif order == "freq_decrease":
                for sample in self.data:
                    sample["output"] = sorted(
                        sample["output"],
                        key=lambda x: self.label_count[x],
                        reverse=True,
                    )
            elif order == "random":
                for sample in self.data:
                    random.shuffle(sample["output"])
            else:
                raise NotImplementedError

            print(
                "Current label order: %s is changed to order: %s, label order rearrangement is completed."
                % (past_order, self.order)
            )

    def save_labels_to_file(
        self, label_save_path
    ):  # TODO check if path exist and file exist
        with PathManager.open(label_save_path, "w") as f:
            f.write(self.label_count)
            print(
                "Label file is saved to %s with label order: %s"
                % (label_save_path, self.order)
            )
        f.close()

    def save_data_to_file(self, data_save_path):
        if data_save_path:  # TODO check if path exist and file exist
            with PathManager.open(data_save_path, "w") as op_fp:
                for sample in self.data:
                    op_fp.write(json.dumps(sample) + "\n")
            logger.info(
                "Data file is saved to %s with label order: %s"
                % (data_save_path, self.order)
            )

    def __len__(self):
        return len(self.data)

    def order_labels(self, label_seq):
        if self.order == "random":
            random.shuffle(label_seq)
            label_seq_ordered = label_seq
        elif self.order == "freq_increase":
            if self.label_count is None:
                self.get_label_count()
            label_seq_ordered = sorted(label_seq, key=lambda x: self.label_count[x])
            return label_seq_ordered
        elif self.order == "freq_decrease":
            if self.label_count is None:
                self.get_label_count()
            label_seq_ordered = sorted(
                label_seq, key=lambda x: self.label_count[x], reverse=True
            )
        else:
            raise NotImplementedError
        return label_seq_ordered

    def label_to_str(self, label):
        return label.replace("_", " ") if self.replace_underscores else label

    def output_str_to_labels(self, output_str):
        return [x.strip() for x in output_str.split(self.sep)]

    def token_ids_to_labels(self, tokenizer, token_ids):
        return self.output_str_to_labels(
            tokenizer.decode(token_ids).split("</s>")[0].replace("<pad>", "")
        )

    def make_example(self, idx):

        assert (
            self.data is not None
        ), "Attempted to access data before loading it. Call read_data() first"

        example = self.data[idx]
        # The following modification is a work around to accommodate all the dataset
        # in Amazon and wiki dataset: the key is "uid", in EUR-lex, the key is "id"
        # id = example["id"] if example.get("id", False) else example.get("uid", None)
        input_text = example["input"]
        labels = (
            self.order_labels(example["output"])
            if self.order == "random"
            else example["output"]
        )

        # if idx == 0:
        #     print(
        #         input.lower().strip(),  # in
        #         self.sep.join(self.label_to_str(label) for label in labels)  # out
        #     )

        out_str = self.sep.join(self.label_to_str(label) for label in labels)

        return (input_text.lower().strip(), out_str)

    def __getitem__(self, idx):
        return self.make_example(idx)

    def return_all_inputs(self):
        qs = []
        for i in range(len(self.data)):
            qs.append(self.make_example(i)[0])
        return qs

    def get_all_labels(self):
        assert (
            self.label_set is not None
        ), "Attempted to access data before loading it. Call read_data() first"
        return {self.label_to_str(label) for label in self.label_set}
