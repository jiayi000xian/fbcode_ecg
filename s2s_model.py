import functools
import logging
import math
import os
from collections import defaultdict
from time import time

import torch
import torch.multiprocessing as mp
from iopath.common.file_io import PathManager as PathManagerBase
from iopath.fb.manifold import ManifoldPathHandler
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    T5ForConditionalGeneration,
)

logger = logging.getLogger(__name__)

PathManager = PathManagerBase()
PathManager.register_handler(
    ManifoldPathHandler(timeout_sec=1800, max_parallel=32), allow_override=True
)

MANIFOLD_BUCKET_NAME = "fast_content_understanding"
MANIFOLD_BUCKET_DIR = "tree/ecg/flow"
MANIFOLD_API_KEY = "fast_content_understanding-key"

RETENTION_DAYS = 90


def compute_metrics(preds, golds):
    metrics = defaultdict(float)

    num_datapoints = len(golds)
    assert len(preds) == num_datapoints

    for g, p in zip(golds, preds):
        g_labels = set(g)
        p_labels = set(p)
        inter = p_labels.intersection(g_labels)

        metrics["micro_accuracy"] += (
            (1.0) * len(inter) / len(p_labels) if len(p_labels) > 0 else 0.0
        )
        metrics["micro_recall"] += (1.0) * len(inter) / len(g_labels)
        for k in [1, 3, 5]:
            topk_inter = set(p[:k]).intersection(g_labels)
            metrics[f"P@{k}"] += (1.0) * len(topk_inter) / k

    return {k: v * 1.0 / num_datapoints for k, v in metrics.items()}


def make_s2s_model(EcgHelper, from_model, from_checkpoint, device):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(EcgHelper.load_model_local_dir)
    logger.info("tokenizer has been loaded from %s" % EcgHelper.load_model_local_dir)

    # load model
    if from_checkpoint:  # TODO
        model = AutoModelForSeq2SeqLM.from_pretrained(
            EcgHelper.load_checkpoint_local_dir
        ).to(device)
    else:
        logger.info("start loading model from %s" % EcgHelper.load_model_local_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            EcgHelper.load_model_local_dir
        ).to(device)
        logger.info("model loading is completed.")

    s2s_optimizer = None
    s2s_scheduler = None
    best_eval = None
    if from_checkpoint:
        param_dict = torch.load(
            os.path.join(EcgHelper.load_checkpoint_local_dir, "training_para.pth"),
            map_location=device,
        )  # param_dict contains weights, optimizer, and scheduler states
        s2s_optimizer = AdamW(model.parameters(), lr=0.0001, eps=1e-8)
        s2s_scheduler = get_linear_schedule_with_warmup(
            s2s_optimizer,
            num_warmup_steps=400,
            num_training_steps=1,
        )
        s2s_optimizer.load_state_dict(param_dict["optimizer"])
        s2s_scheduler.load_state_dict(param_dict["scheduler"])
        if "loss_spearman" in param_dict["best_eval"]:
            best_eval = param_dict["best_eval"]["loss_spearman"]
        else:
            best_eval = param_dict["best_eval"]["loss"]

    return s2s_scheduler, s2s_optimizer, tokenizer, model, best_eval


def make_s2s_batch(
    io_list,
    tokenizer,
    model,
    label_trie=None,
    max_i_len=512,
    max_o_len=16,
    device="cuda",
):
    if device and device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i_ls = [i for i, o in io_list]
    o_ls = [o for i, o in io_list]

    i_toks = tokenizer(
        i_ls, max_length=max_i_len, padding="max_length", truncation=True
    )
    i_ids, i_mask = (
        torch.LongTensor(i_toks["input_ids"]).to(device),
        torch.LongTensor(i_toks["attention_mask"]).to(device),
    )

    # source_len = [(q_ids[:, x-1] == tokenizer.pad_token_id).sum().item() for x in [128, 256, 512]] + [q_ids.shape[0]]
    # tokenizer.source_len = [x + y for x, y in zip(tokenizer.source_len, source_len)]

    o_toks = tokenizer(
        o_ls, max_length=max_o_len + 1, padding="max_length", truncation=True
    )
    # for tokenized_text in o_toks["input_ids"]:
    #     assert tokenized_text[-1] == tokenizer.pad_token_id, "We're truncating labels. This will lead to invalid labels!"

    o_ids, _ = (
        torch.LongTensor(o_toks["input_ids"]).to(device),
        torch.LongTensor(o_toks["attention_mask"]).to(device),
    )

    # I don't know why the input for t5 and bart is different, copy-pasted from HF examples
    if isinstance(model, DataParallel):
        model = model.module

    if isinstance(model, T5ForConditionalGeneration):
        decoder_input_ids = model._shift_right(o_ids)
        lm_labels = o_ids
    else:
        decoder_input_ids = o_ids[:, :-1].contiguous()
        lm_labels = o_ids[:, 1:].contiguous().clone()

    model_inputs = {
        "input_ids": i_ids,
        "attention_mask": i_mask,
        "decoder_input_ids": decoder_input_ids,
        "labels": lm_labels,
    }

    # Compute target for multi-option loss
    if label_trie:
        model_inputs["targets"] = label_trie.compute_targets(o_ids, o_ls)

    return model_inputs


def train_s2s_epoch(
    EcgHelper,
    model,
    dataset,
    tokenizer,
    label_trie,
    optimizer,
    scheduler,
    args,
    e=0,
    curriculum=False,
):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)

    tokenizer.source_len = [0.0, 0.0, 0.0, 0.0]
    if args.device and args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_collate_fn = functools.partial(
        make_s2s_batch,
        model=model,
        tokenizer=tokenizer,
        label_trie=label_trie,
        max_i_len=args.max_i_length,
        max_o_len=args.max_o_length,
        device=args.device,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=model_collate_fn,
    )
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)

    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):

        if args.use_multisoftmax:
            # Targets matrix that allows for all possible next tokens at a given time
            # Dimension: batch_size x max_output_len x num_tokens
            # Need to pass label_trie to model_collate_fn for this to work
            # Passing this as part of batch_inputs is a hack. The model doesn't actually expect this input so need to pop
            targets = batch_inputs.pop("targets", None).to(device)

        model_output = model(**batch_inputs)

        if args.use_multisoftmax:
            # Variaion on SoftMax that'll allow of to distribute weight over different outcomes

            # We'll compute the loss ourselves outside of the model. We need the raw logits for that.
            lm_logits = model_output[1]

            # We want to take the sum of the exp() of the logits correspodinging to possible next tokens.
            # For everything else we want 0, this is achieved by exp(-inf)
            gt_label_logits = lm_logits.masked_fill(targets == 0, float("-inf"))

            # Intuitively, this removes competition between correct labels
            # torch.logsumexp(gt_label_logits, dim=2) is just the multi-otion equivalent of the
            # log(exp(gt_label_logit)) term in the single label case.
            multisoftmax_per_token = -torch.logsumexp(
                gt_label_logits, dim=2
            ) + torch.logsumexp(lm_logits, dim=2)

            pre_loss = torch.mean(multisoftmax_per_token)
        else:
            # Use the vanilla softmax of the HF model of choice
            pre_loss = model_output[0]

        loss = pre_loss
        loss.backward()
        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            logger.info(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e,
                    step,
                    len(dataset) // args.train_batch_size,
                    loc_loss / loc_steps,
                    time() - st_time,
                )
            )
            output_train_file = os.path.join(EcgHelper.local_dir, "output_train")
            with open(output_train_file, "a+") as dev_file:
                dev_file.write(
                    "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}\n".format(
                        e,
                        step,
                        len(dataset) // args.train_batch_size,
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )
            loc_loss = 0
            loc_steps = 0

    # print("\n\nTrain stats:\n<128\t<256\t<512:", tokenizer.source_len, "\n")


def eval_s2s_epoch(EcgHelper, model, dataset, tokenizer, args, epoch):
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
    st_time = time()

    with torch.no_grad():
        preds = []
        golds = []
        for step, batch_inputs in enumerate(epoch_iterator):
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
                num_beams=1,
                temperature=1.0,
                top_k=None,
                top_p=None,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                decoder_start_token_id=tokenizer.bos_token_id,
            )

            generated_ids = list(generated_ids)

            raw_preds = [
                (tokenizer.decode(ans_ids).split("</s>")[0].replace("<pad>", ""))
                for ans_ids in generated_ids
            ]

            pred = [dataset.output_str_to_labels(pred) for pred in raw_preds]

            gold = [
                dataset.output_str_to_labels(
                    tokenizer.decode(ans_ids).split("</s>")[0].replace("<pad>", "")
                )
                for ans_ids in batch_inputs["labels"]
            ]

            # Print to quickly debug predictions
            # print(generated_ids[0])
            # print("raw_pred", raw_preds[0])
            # print("pred", pred[0])
            # print("gold", gold[0])

            golds.extend(gold)
            preds.extend(pred)

            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                logger.info(
                    "{:5d} of {:5d} \t L: {:.6f} \t  -- {:.3f}".format(
                        step,
                        len(dataset) // args.eval_batch_size,
                        loc_loss / loc_steps,
                        time() - st_time,
                    )
                )

    predictions_file = os.path.join(EcgHelper.local_dir, "predictions_dev")
    with open(predictions_file, "a") as dev_file:
        dev_file.write(f"epoch {epoch}" + "\n")
        idx = list(range(len(golds)))
        for i, g, p in zip(idx, golds, preds):
            dev_file.write(str(i) + "\t" + str(g) + "\t" + str(p) + "\n")

    metrics = compute_metrics(
        preds, golds
    )  # TODO add more metrics and recall@precision
    metric_str = "L: {:.3f} ".format(loc_loss / loc_steps) + " ".join(
        f"{k}: {v:.3f}" for k, v in metrics.items()
    )

    output_dev_file = os.path.join(EcgHelper.local_dir, "output_dev")
    with open(output_dev_file, "a") as dev_file:
        dev_file.write(f"epoch {epoch}" + "\t" + metric_str + "\n")
    logger.info(metric_str)

    if epoch % args.output_upload_interval == 0:
        EcgHelper.save_training_result_manifold(
            ["predictions_dev", "output_dev", "output_train"]
        )

    return loc_loss, metrics["P@3"]  # Use P@3 to decide best model


def train_s2s_parallel(
    EcgHelper,
    s2s_model,
    s2s_tokenizer,
    label_trie,
    s2s_optimizer,
    s2s_scheduler,
    best_eval,
    s2s_train_dset,
    s2s_valid_dset,
    s2s_args,
):
    if s2s_optimizer is None:
        s2s_optimizer = AdamW(
            s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8
        )
    if s2s_scheduler is None:
        s2s_scheduler = get_linear_schedule_with_warmup(
            s2s_optimizer,
            num_warmup_steps=400,
            num_training_steps=(s2s_args.num_epochs + 1)
            * math.ceil(len(s2s_train_dset) / s2s_args.train_batch_size),
        )
    for e in range(s2s_args.num_epochs):
        train_s2s_epoch(
            EcgHelper,
            s2s_model,
            s2s_train_dset,
            s2s_tokenizer,
            label_trie,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )

        # Decoding can be slow, we can control how often we want to do that
        if e % s2s_args.eval_every_k_epoch == s2s_args.eval_every_k_epoch - 1:
            eval_l, eval_acc = eval_s2s_epoch(
                EcgHelper, s2s_model, s2s_valid_dset, s2s_tokenizer, s2s_args, e
            )

            if best_eval is None or eval_acc > best_eval:
                best_eval = eval_acc
                if s2s_args.checkpoint_save_option:

                    m_save_dict = {
                        "optimizer": s2s_optimizer.state_dict(),
                        "scheduler": s2s_scheduler.state_dict(),
                        "best_eval": {"em": eval_acc},
                    }

                    # saving on epoch with current best evaluation (without overwrite)
                    """
                    local: ${EcgHelper.local_dir}/checkpoint/${saving_suffix}/pytorch_model.bin, configs.json, training_para.pth
                    remote: ${EcgHelper.flow_dir}/checkpoint/${saving_suffix}/pytorch_model.bin, configs.json, training_para.pth
                    """
                    if s2s_args.checkpoint_save_option == "eval":
                        saving_suffix = s2s_args.model_save_name + "_epoch_" + str(e)
                    # saving on epoch with current best evaluation (overwrite)
                    elif s2s_args.checkpoint_save_option == "best":
                        saving_suffix = s2s_args.model_save_name + "_best"
                    else:
                        raise NotImplementedError

                    local_checkpt_dir = os.path.join(
                        EcgHelper.local_dir, "checkpoint", saving_suffix
                    )
                    logger.info("Saving model locally to {}".format(local_checkpt_dir))

                    if isinstance(s2s_model, DataParallel):
                        s2s_model.module.save_pretrained(local_checkpt_dir)
                    else:
                        s2s_model.save_pretrained(local_checkpt_dir)

                    torch.save(
                        m_save_dict,
                        os.path.join(local_checkpt_dir, "training_para.pth"),
                    )

                    if e % s2s_args.checkpoint_upload_interval == 0:
                        EcgHelper.save_checkpoint_manifold(saving_suffix)


def train_s2s(
    EcgHelper,
    s2s_model,
    s2s_tokenizer,
    label_trie,
    s2s_optimizer,
    s2s_scheduler,
    best_eval,
    s2s_train_dset,
    s2s_valid_dset,
    s2s_args,
):
    if s2s_args.num_epochs == 0:
        return
    model_name = s2s_args.model_name_or_path.lower()

    if "led" in model_name:
        raise NotImplementedError
    else:
        # if not isinstance(s2s_model, DataParallel):
        #    model = torch.nn.DataParallel(s2s_model, device_ids=[i for i in range(s2s_args.n_gpu)])
        # else:
        #    model = s2s_model

        train_s2s_parallel(
            EcgHelper,
            s2s_model,
            s2s_tokenizer,
            label_trie,
            s2s_optimizer,
            s2s_scheduler,
            best_eval,
            s2s_train_dset,
            s2s_valid_dset,
            s2s_args,
        )


# generate answer from input "input: ... context: <p> ..."
def s2s_generate_v1(
    question_docs,
    s2s_model,
    s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=1,
    max_len=32,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cuda",
):
    model_inputs = make_s2s_batch(
        list(zip(question_docs, ["A"] * len(question_docs))),
        s2s_tokenizer,
        max_input_length,
        device=device,
    )
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    answers = []

    generated_ids = s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=s2s_tokenizer.bos_token_id,
    )
    answers = s2s_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return answers


def clone_generate(
    nproc, model, tokenizer, eval_batch_size, queries, max_input_length, clone_save_path
):
    model.to(torch.device(f"cuda:{clone_save_path[-5]}"))
    device = model.device

    predictions = {}
    dataloader = torch.utils.data.DataLoader(
        queries, batch_size=eval_batch_size, shuffle=False
    )
    with torch.no_grad():
        for q_batch in tqdm(dataloader):
            predictions_batch = s2s_generate_v1(
                q_batch[1],
                model,
                tokenizer,
                max_len=32,
                max_input_length=max_input_length,
                device=device,
            )

            for idx, pr in zip(q_batch[0], predictions_batch):
                predictions[idx] = pr

    with open(clone_save_path, "w") as f_out:
        for q in queries:
            f_out.write(f"{q[0]}\t{predictions[q[0]]}\n")


def parallel_generate(
    model,
    tokenizer,
    dataset,
    batch_size=20,
    n_gpu=1,
    max_input_length=512,
    model_path="reader",
):
    # # First GPU has 25% less space
    # span_size = len(dataset) / (n_gpu - 0.25)
    # span_diff = int(span_size * 0.25)
    span_size = len(dataset) / n_gpu

    model.to(torch.device("cpu"))
    model.eval()

    clone_ctx = []
    clone_predicts_path = []
    for i in range(n_gpu):
        clone_save_path = model_path + f"_predictions_clone_{i}.txt"
        clone_predicts_path.append(clone_save_path)

        span = (int(span_size * i), min(int(span_size * (i + 1)), len(dataset)))
        # span = (max(int(span_size * i) - span_diff, 0), min(int(span_size * (i+1)) - span_diff, len(dataset)))

        batch_size_mult = 1
        # if i == 0:
        #     batch_size_mult = 0.75

        clone_args = (
            model,
            tokenizer,
            int(batch_size * batch_size_mult),
            dataset[span[0] : span[1]],
            max_input_length,
            clone_save_path,
        )
        ctx = mp.spawn(clone_generate, clone_args, nprocs=1, join=False)
        clone_ctx.append(ctx)

    for ctx in clone_ctx:
        ctx.join()

    predictions = {}
    for clone_path in clone_predicts_path:
        with open(clone_path, "r") as fin:
            for line in fin:
                line = line.split("\t")
                predictions[line[0]] = line[1].strip()

    return predictions


def s2s_generate_v2(model, tokenizer, dataset, s2s_args):
    model_name = s2s_args.model_name_or_path.lower()
    if "led" in model_name:
        return parallel_generate(
            model,
            tokenizer,
            dataset,
            16,
            s2s_args.n_gpu,
            s2s_args.max_i_length,
            s2s_args.model_save_name,
        )
    else:
        return parallel_generate(
            model,
            tokenizer,
            dataset,
            80,
            s2s_args.n_gpu,
            s2s_args.max_i_length,
            s2s_args.model_save_name,
        )
