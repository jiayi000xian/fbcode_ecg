import logging
import os

import fblearner.flow.api as flow
import torch
from fblearner.flow.api import types

from .common import EcgManifoldHelper
from .data_utils import Seq2SetDataset
from .decode_utils import LabelTrie
from .local_configs import enable_proxy
from .params import DATA_ARGS_SCHEME
from .s2s_model import make_s2s_model, train_s2s
from .utils import prepare_tokenizer

logger = logging.getLogger(__name__)


@flow.flow_async(
    use_resource_prediction=False,
    resource_requirements=flow.ResourceRequirements(cpu=4, gpu=1, memory="20g"),
)
@flow.registered(owners=["jiayi000xian"])
@flow.typed(
    input_schema=types.Schema([("args", DATA_ARGS_SCHEME)]), returns=types.STRING
)
def finetune_s2s(args):

    """
    # for debug use:
    from .params import ArgumentsS2S
    parser = ArgumentsS2S()
    parser.add_train_and_eval_args()
    parser.add_s2s_args()
    s2s_args = parser.parse_args()
    """
    s2s_args = args
    logger.info(s2s_args)

    # create local training data save dir
    EcgHelper = EcgManifoldHelper(
        "training",
        s2s_args.data_path,
        s2s_args.model_name_or_path,
        s2s_args.from_checkpoint,
    )

    # set device
    if s2s_args.device and s2s_args.device.lower() == "cpu":
        true_device = "cpu"
    else:
        true_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(true_device)
    if s2s_args.device.lower() != true_device:
        logger.warn(
            f"Required to train on {s2s_args.device.lower()}, but actually train on {true_device}"
        )

    # download model
    from_checkpoint, from_model = False, False
    if s2s_args.from_checkpoint:
        EcgHelper.load_checkpoint_local_dir = EcgHelper.get_resource_dir_local(
            s2s_args.from_checkpoint
        )
        from_checkpoint = True

    if s2s_args.model_name_or_path:
        EcgHelper.load_model_local_dir = EcgHelper.get_resource_dir_local(
            s2s_args.model_name_or_path
        )
        from_model = True
    else:
        raise NotImplementedError

    s2s_scheduler, s2s_optimizer, tokenizer, model, best_eval = make_s2s_model(
        EcgHelper,
        from_model,
        from_checkpoint,
        device,
    )

    if s2s_args.use_proxy:
        enable_proxy()

    # Construct trie used for decoding and multi-option loss
    prepare_tokenizer(tokenizer)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    # Prepare datasets
    logger.info("starting loading data")
    train_data_path = os.path.join(s2s_args.data_path, s2s_args.train_file)
    test_data_path = os.path.join(s2s_args.data_path, s2s_args.test_file)
    logger.info(
        f"Training and testing data has been loaded from {train_data_path} and {test_data_path}"
    )

    # valid_data = os.path.join(s2s_args.data_path, s2s_args.valid_file)
    label_order = s2s_args.label_order
    s2s_train_set = Seq2SetDataset(
        train_data_path,
        label_order,
        sep_token,
        replace_underscores=s2s_args.replace_underscores,
        read_per_line=s2s_args.read_per_line,
    )
    s2s_dev_set = Seq2SetDataset(
        test_data_path,
        label_order,
        sep_token,
        replace_underscores=s2s_args.replace_underscores,
        read_per_line=s2s_args.read_per_line,
    )

    if s2s_args.use_multisoftmax:
        # To allow for any possible next token at a given time, we need a label trie
        # that will compute the corresponding target tensors.
        label_trie = LabelTrie.from_labels(
            s2s_train_set.get_all_labels().union(s2s_dev_set.get_all_labels()),
            tokenizer,
            sep_token,
        )
    else:
        label_trie = None

    # for using on multiple gpu using DataParallel
    if s2s_args.data_parallel:
        s2s_model = torch.nn.DataParallel(model)
    else:
        s2s_model = model

    train_s2s(
        EcgHelper,
        s2s_model,
        tokenizer,
        label_trie,
        s2s_optimizer,
        s2s_scheduler,
        best_eval,
        s2s_train_set,
        s2s_dev_set,
        s2s_args,
    )
    return str(EcgHelper.flow_dir)


"""
if __name__ == "__main__":
    finetune_s2s()
"""
