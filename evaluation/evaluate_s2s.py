# fmt: off

# To apply the fixes over an existing project, run:
# arc lint --apply-patches --take BLACK --paths-cmd 'hg files -I "your-project/**/*.py"'

# isort:skip
import functools
import json
import logging
import os
import random
from collections import defaultdict

import fblearner.flow.api as flow
import torch
from fblearner.flow.api import types
from fblearner.flow.projects.ecg.common import EcgManifoldHelper
from fblearner.flow.projects.ecg.data_utils import Seq2SetDataset
from fblearner.flow.projects.ecg.decode_utils import score_labels_by_probability_sum, LabelTrie
from fblearner.flow.projects.ecg.s2s_model import make_s2s_batch
from fblearner.flow.projects.ecg.utils import prepare_tokenizer
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from .gen_curve import gen_curve, compute_metrics
from .params import EVAL_ARGS_SCHEME  # ArgumentsS2S for debug


logger = logging.getLogger(__name__)
# enable_proxy()


# A version of eval_s2s_epoch but with more excessive logging of interesting stats
def decode_s2s(model, EcgHelper, dataset, label_set, tokenizer, args, sep_token):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_s2s_batch,
        model=model,
        tokenizer=tokenizer,
        max_i_len=args.max_i_length,
        max_o_len=args.max_o_length,
        device=args.device,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        sampler=train_sampler,
        collate_fn=model_collate_fn,
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0

    if args.decode_on_lattice:
        # Build trie of all possible labels
        label_trie = LabelTrie.from_labels(label_set, tokenizer, sep_token)

    def decode_on_label_lattice(batch_id, input_ids):
        next_tokens, completed_label_ids = label_trie.next_allowed_token(
            input_ids.tolist()[1:]
        )

        # Uncomment this to debug the decoding process
        # if batch_id == 0:
        #     input_tokens = tokenizer.decode(input_ids[1:], skip_special_tokens=False)
        #     print(" " * len (input_ids), input_ids.tolist()[1:], f'"{input_tokens}"', next_tokens if len(next_tokens) < 100 else "all tokens")

        return next_tokens

    with torch.no_grad():
        preds_by_method = defaultdict(list)
        golds = []
        for step, batch_inputs in enumerate(epoch_iterator):

            example_ids = batch_inputs["input_ids"]
            #del batch_inputs["example_ids"]

            pre_loss = model(**batch_inputs)[0]

            if isinstance(model, DataParallel):
                model_gen = model.module
                loss = pre_loss.sum() / pre_loss.shape[0]
            else:
                model_gen = model
                loss = pre_loss

            generated_ids = model_gen.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                min_length=1,
                max_length=args.max_o_length + 1,
                do_sample=False,
                early_stopping=True,
                num_beams=args.decode_beams,
                temperature=1.0,
                top_k=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=args.decode_beams,
                decoder_start_token_id=tokenizer.bos_token_id,
                prefix_allowed_tokens_fn=decode_on_label_lattice
                if args.decode_on_lattice
                else None,
                return_dict_in_generate=True,
                output_scores=True,
            )

            if args.decode_beams > 1:
                # Use beam search to find most likely sequences and integrate over those sequences

                # Decoder doesn't have a separate dimension for beam size, need to reshape.
                # num_examples might be less than args.eval_batch_size in the last batch
                num_examples = generated_ids["sequences"].size()[0] // args.decode_beams
                for example_id, sequences, scores, labels in zip(
                    example_ids,
                    generated_ids["sequences"].view(
                        num_examples, args.decode_beams, -1
                    ),
                    generated_ids["sequences_scores"].view(
                        num_examples, args.decode_beams, 1
                    ),
                    batch_inputs["labels"],
                ):
                    top_label_sequences = []

                    for sequence in sequences:  # num_beams * bsz
                        top_label_sequences.append(
                            dataset.token_ids_to_labels(tokenizer, sequence)
                        )  # decode the sequences

                    naive_preds = top_label_sequences[0]  # the seq with highest prob (log prob)
                    filtered_preds = [label for label in naive_preds if label in label_set]
                    sum_prob_preds = score_labels_by_probability_sum(
                        top_label_sequences,
                        scores,
                    )
                    filtered_sum_prob_preds = [
                        label for label in sum_prob_preds if label in label_set
                    ]

                    preds_by_method["naive"].append((example_id, naive_preds))
                    preds_by_method["filtered"].append((example_id, filtered_preds))
                    preds_by_method["sum_prob"].append((example_id, sum_prob_preds))
                    preds_by_method["filtered_sum_prob"].append(
                        (example_id, filtered_sum_prob_preds)
                    )
                    golds.append(dataset.token_ids_to_labels(tokenizer, labels))

                    # print("Gold labels: ", golds[-1])
                    # print("Top beams:")
                    # for score, sequence in zip(scores, top_label_sequences):
                    #     print(score.item(), sequence)
                    # print("All of the above ordered by sum probability:", sum_prob_preds)
                    # print("The above but filtered to valid labels:", filtered_sum_prob_preds)
                    # print("\n\n")
            else:
                # No beam search, work with the simple greedy output sequence
                for example_id, output_token_ids, label_token_ids in zip(
                    example_ids, generated_ids["sequences"], batch_inputs["labels"]
                ):

                    naive_preds = dataset.token_ids_to_labels(
                        tokenizer, output_token_ids
                    )
                    filtered_preds = [label for label in naive_preds if label in label_set]
                    gold = dataset.token_ids_to_labels(tokenizer, label_token_ids)

                    # print(naive_preds)
                    # print(filtered_preds)
                    # print(gold)
                    # print("\n")

                    preds_by_method["naive"].append((example_id, naive_preds))
                    preds_by_method["filtered"].append((example_id, filtered_preds))
                    golds.append(gold)

            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d}".format(step, len(dataset) // args.eval_batch_size)
                )

            # For the impatient kind
            # if len(golds) % 100 == 0:
            #     for method, preds in preds_by_method.items():
            #         metrics = compute_metrics(preds, golds)
            #         print(f"  {method}: " + " ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))

    print("Loss: {:.3f}".format(loc_loss / loc_steps))

    return preds_by_method, golds


def save_preds_and_golds(preds_by_method, golds, EcgHelper, args):
    for method, preds in preds_by_method.items():

        # debug setting: EcgHelper.local_dir = "/data/sandcastle/boxes"
        method_str = method + ("_lattice" if args.decode_on_lattice else "")
        preds_file = os.path.join(
            EcgHelper.local_dir, f"test_preds_{method_str}.jsonl"
        )

        with open(preds_file, "w") as outfile:
            for _, pred in preds:
                outfile.write(json.dumps({"output": pred}) + "\n")

    golds_file = os.path.join(EcgHelper.local_dir, "test_golds.json")

    with open(golds_file, "w") as outfile:
        json.dump(golds, outfile)


def metrics_post_processing(preds_by_method, golds, EcgHelper, args, get_curve: str = "all"):

    if get_curve != "curve_only":

        for method, preds in preds_by_method.items():

            if method == "sum_prob":
                # for preds obtained by "sum_prob" method, each item in preds is in the shape ( id, dict of {label : score})
                preds_with_scores = [x[1] for x in preds]  # list of dict of {label : score}
                metrics = compute_metrics([x[0] for x in preds_with_scores.items()], golds)
            else:
                metrics = compute_metrics([x[1] for x in preds], golds)  # preds (sum_prob) : bsz * 2 bsz * (id, {label:score})

            logger.info(f"{method}: " + " ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
            method_str = method + ("_lattice" if args.decode_on_lattice else "")

            metrics_file = os.path.join(
                EcgHelper.local_dir, f"test_metrics_{method_str}.json"
            )
            with open(metrics_file, "w") as outfile:
                json.dump(metrics, outfile)

    if get_curve:

        method = "curve_sum_prob"
        metrics, _ = gen_curve(preds_by_method["sum_prob"], golds, args.num_thresholds, args.min_support)

        logger.info(f"{method}: " + " ".join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        method_str = method + ("_lattice" if args.decode_on_lattice else "")

        metrics_file = os.path.join(
            EcgHelper.local_dir, f"test_metrics_{method_str}.json"
        )
        with open(metrics_file, "w") as outfile:
            json.dump(metrics, outfile)


def show_examples(preds_by_method, golds, label_set):
    # Show stats and examples for the raw model output
    preds = [x[1] for x in preds_by_method["naive"]]

    preds_at_pos = defaultdict(int)
    oov_samples_by_pos = defaultdict(list)
    for p, g in zip(preds, golds):
        for pos, label in enumerate(p):
            preds_at_pos[pos] += 1
            if label not in label_set:
                oov_samples_by_pos[pos].append((label, p, g))

    for pos in range(5):
        logger.info(
            f"\n *** OOV rate at position {pos}: {len(oov_samples_by_pos[pos]) / preds_at_pos[pos] if preds_at_pos[pos] != 0 else 0} ***"
        )
        if len(oov_samples_by_pos[pos]) >= 5:
            for oov_pred, prediction, gold_labels in random.sample(
                oov_samples_by_pos[pos], 5
            ):
                logger.info(f"\n  OOV prediction: {oov_pred}")
                logger.info(f"\n  Predicted: {prediction}")
                logger.info(f"\n  Gold: {gold_labels}")


@flow.flow_async(
    use_resource_prediction=False,
    resource_requirements=flow.ResourceRequirements(cpu=4, gpu=1, memory="10g"),
)
@flow.registered(owners=["jiayi000xian"])
@flow.typed(
    input_schema=types.Schema([("args", EVAL_ARGS_SCHEME)]), returns=types.STRING
)
def evaluate(args):

    """
    # uncommon this for debug:
    parser = ArgumentsS2S()
    parser.add_train_and_eval_args()
    parser.add_s2s_args()
    eval_args = parser.parse_args()
    """

    eval_args = args
    logger.info(eval_args)

    EcgHelper = EcgManifoldHelper(
        "eval", eval_args.data_path, eval_args.model_name_or_path, eval_args.from_checkpoint
    )

    # set device
    if eval_args.device and eval_args.device.lower() == "cpu":
        true_device = "cpu"
    else:
        true_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(true_device)
    if eval_args.device.lower() != true_device:
        logger.warn(
            f"Required to train on {eval_args.device.lower()}, but actually train on {true_device}"
        )
    # download model
    EcgHelper.load_model_local_dir = EcgHelper.get_resource_dir_local(eval_args.model_name_or_path)
    if eval_args.from_checkpoint:
        EcgHelper.load_checkpoint_local_dir = EcgHelper.get_resource_dir_local(eval_args.from_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(EcgHelper.load_model_local_dir)
    logger.info("tokenizer has been loaded from %s" % EcgHelper.load_model_local_dir)
    prepare_tokenizer(tokenizer)
    sep_token = tokenizer.sep_token if tokenizer.sep_token else "[SEP]"

    print("sep token: ", sep_token)

    if eval_args.from_checkpoint:  # TODO
        model = AutoModelForSeq2SeqLM.from_pretrained(
            EcgHelper.load_checkpoint_local_dir
        ).to(device)
    elif eval_args.model_name_or_path:
        logger.info("start loading model from %s" % EcgHelper.load_model_local_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            EcgHelper.load_model_local_dir
        ).to(device)
        logger.info("model loading is completed.")
    else:
        raise NotImplementedError

        # Prepare datasets
    logger.info("starting loading data")
    train_data_path = os.path.join(eval_args.data_path, eval_args.train_file)
    test_data_path = os.path.join(eval_args.data_path, eval_args.test_file)
    logger.info(
        f"Training and testing data has been loaded from {train_data_path} and {test_data_path}"
    )

    label_order = eval_args.label_order
    s2s_train_set = Seq2SetDataset(
        train_data_path,
        label_order,
        sep_token,
        replace_underscores=eval_args.replace_underscores,
        read_per_line=eval_args.read_per_line,
    )
    s2s_dev_set = Seq2SetDataset(
        test_data_path,
        label_order,
        sep_token,
        replace_underscores=eval_args.replace_underscores,
        read_per_line=eval_args.read_per_line,
    )

    train_label_set = s2s_train_set.get_all_labels()
    dev_label_set = s2s_dev_set.get_all_labels()
    print("# of distinct labels in train set:", len(train_label_set))
    print("# of distinct labels in dev set:", len(dev_label_set))
    print("# of new labels in dev set:", len(dev_label_set.difference(train_label_set)))
    all_labels_set = train_label_set.union(dev_label_set)

    preds_by_method, golds = decode_s2s(model, EcgHelper, s2s_dev_set, all_labels_set, tokenizer, eval_args, sep_token)

    save_preds_and_golds(preds_by_method, golds, EcgHelper, eval_args)

    metrics_post_processing(preds_by_method, golds, EcgHelper, eval_args, get_curve=eval_args.get_curve)

    if eval_args.show_examples:
        show_examples(preds_by_method, golds, all_labels_set)

    # upload to manifold
    output_files = os.listdir(EcgHelper.local_dir)
    EcgHelper.save_training_result_manifold(output_files)

    return str(EcgHelper.flow_dir)

# fmt: off
