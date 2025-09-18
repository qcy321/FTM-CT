'''
    @Project ：
    @File    ：train.py.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/25 21:54
    @Description :
'''
import json
import logging
import os
import time
from typing import Optional

import torch
import numpy as np
from torch import optim

from torch.cuda.amp import autocast, GradScaler

from utils import contrastive_loss, update_momentum_encoder
from dataset import TextDataset
from common import CheckpointType, FinetuneStrategy, ModelClass, SaveModelFileName

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)


class TrainingComplete(Exception):
    """Custom exception used to jump out of the training loop"""
    pass


def setup_optimizer_scheduler(args, model, num_training_steps: int):
    """Set the optimizer and learning rate scheduler."""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Load the saved optimizer and scheduler state
    last_path = os.path.join(args.output_dir, args.model_name, CheckpointType.LAST_MRR.value)
    try:
        if os.path.exists(os.path.join(last_path, 'optimizer.pt')):
            optimizer.load_state_dict(torch.load(os.path.join(last_path, 'optimizer.pt', weights_only=False)))
        if os.path.exists(os.path.join(last_path, 'scheduler.pt')):
            scheduler.load_state_dict(torch.load(os.path.join(last_path, 'scheduler.pt', weights_only=False)))
    except Exception as e:
        logger.warning(f"Failed to load the optimizer or scheduler: {e}")

    return optimizer, scheduler


def train_epoch(args, model, tokenizer, train_dataloader, optimizer, scheduler, scaler,
                epoch_idx: int, cur_epoch=None, dup=None) -> Optional[int]:
    """Train the model for one epoch."""
    model.train()
    tr_loss, tr_num = 0, 0
    for step, batch in enumerate(train_dataloader):
        nl_inputs = batch[0].to(args.device)
        code_inputs = batch[1].to(args.device)
        nl_mask = batch[2].to(args.device) if args.model_class == ModelClass.GRAPH.value else nl_inputs.ne(
            tokenizer.pad_token_id)
        code_mask = code_inputs.ne(tokenizer.pad_token_id)
        ids = batch[3].to(args.device) if args.model_class == ModelClass.GRAPH.value else None

        with autocast(enabled=args.fp16):
            q = model(nl_inputs=nl_inputs, attention_mask=nl_mask, position_ids=ids)
            k = args.strategy.execute(args, epoch_idx, model, code_inputs, code_mask, ids)
            loss = contrastive_loss(q, k, args,
                                    args.temperature if args.hidden_state_method == "avg" else 1)

        tr_loss += loss.item()
        tr_num += 1

        if args.train_mode == "pretrain":
            if (args.global_step + 1) % args.log_interval == 0:
                logger.info(
                    "epoch-{}-dup-{}-cur_epoch-{} step {} loss {}".format(epoch_idx, dup, cur_epoch,
                                                                          args.global_step + 1,
                                                                          round(tr_loss / tr_num, 5)))
                tr_loss = 0
                tr_num = 0
        else:
            if (step + 1) % args.log_interval == 0:
                logger.info(f"epoch {epoch_idx} step {step + 1} loss {round(tr_loss / tr_num, 5)}")
                tr_loss, tr_num = 0, 0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        if args.do_momentum and args.strategy.manager.in_strategy:
            update_momentum_encoder(model, args.strategy.manager.k_encoder)

        if args.train_mode == "pretrain":
            args.global_step += 1
            if args.global_step >= args.max_steps:
                raise TrainingComplete()


def pre_training(args, model, tokenizer):
    """ Train the model """

    # get optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(args, model, args.max_steps)
    scaler = GradScaler(enabled=args.fp16)  # enabled=args.fp16 ensures FP16 is only enabled when needed

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size * max(1, args.n_gpu))
    logger.info("  Total optimization steps = %d", args.max_steps)

    model.zero_grad()
    best_mrr = args.best_mrr

    idx = args.idx
    start_idx = args.start_idx
    dup = args.start_dup

    start_index = (idx // args.num_train_epochs) % len(args.all_lang)
    try:
        while args.max_steps > 0 and args.global_step <= args.max_steps:
            for index in range(start_index, len(args.all_lang)):
                args.train_data_file = args.train_data_files[index]
                args.eval_data_file = args.eval_data_files[index]
                args.test_data_file = args.test_data_files[index]
                args.codebase_file = args.codebase_files[index]
                cur_lang = args.all_lang[index]
                args.output_dir = os.path.join(args.root_output_dir, cur_lang)

                train_dataset = TextDataset(tokenizer, args, args.train_data_file)
                train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                              batch_size=args.train_batch_size * max(1, args.n_gpu), num_workers=4)
                if start_idx == 0:
                    args.strategy.cache.reset()

                logger.info(f"***** {cur_lang} *****")
                logger.info("  Num examples = %d", len(train_dataset))
                logger.info("  Num Epochs = %d", args.num_train_epochs)
                logger.info("  per optimization steps = %d", len(train_dataloader))
                logger.info("  Cur lang total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)
                for epoch in range(start_idx, args.num_train_epochs):
                    train_epoch(args, model, tokenizer, train_dataloader, optimizer,
                                scheduler, scaler, idx, epoch, dup)

                    save_last_model(args, model, optimizer, scheduler, idx, args.root_output_dir)
                    # evaluate
                    results = evaluate(args, model, tokenizer, args.eval_data_file)
                    logger.info("Evaluation results:")
                    for key, value in results.items():
                        logger.info("  %s = %s", key, value)
                    with open(args.mrr_result, "a") as f:
                        f.write(json.dumps(results) + "\n")

                    # save best model
                    output_dir, best_mrr[index] = save_best_model(args, results['eval_mrr'], best_mrr[index], model,
                                                                  results)
                    if output_dir is not None:
                        torch.save(best_mrr, os.path.join(output_dir, "best.pt"))
                        logger.info(f"Saving {best_mrr} checkpoint to {output_dir}/best.pt")

                    idx += 1

                start_idx = 0

            if args.global_step < args.max_steps:
                per_dir = os.path.join(args.root_output_dir, args.model_name, CheckpointType.PER.value, str(dup))
                save_model(model, per_dir, True)
                logger.info(f"Save the {dup}-th round model checkpoint to {per_dir}")
                dup += 1
                start_index = 0

    except TrainingComplete:
        pass

    final_dir = os.path.join(args.root_output_dir, args.model_name, CheckpointType.FINAL.value)
    save_model(model, final_dir, True)
    logger.info("Saving model final checkpoint to %s", final_dir)


def fine_tuning(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size * max(1, args.n_gpu), num_workers=4)

    num_training_steps = len(train_dataloader) * args.num_train_epochs

    optimizer, scheduler = setup_optimizer_scheduler(args, model, num_training_steps)
    # Initialize GradScaler for FP16
    scaler = GradScaler(enabled=args.fp16)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    logger.info("  Total train batch size  = %d", args.train_batch_size * max(1, args.n_gpu))
    logger.info("  per optimization steps = %d", len(train_dataloader))
    logger.info("  Total optimization steps = %d", num_training_steps)

    model.zero_grad()

    best_mrr = args.best_mrr
    for idx in range(args.start_idx, args.num_train_epochs):
        train_epoch(args, model, tokenizer, train_dataloader, optimizer, scheduler, scaler, idx)
        save_last_model(args, model, optimizer, scheduler, idx)

        # evaluate
        results = evaluate(args, model, tokenizer, args.eval_data_file)
        logger.info("Evaluation results:")
        for key, value in results.items():
            logger.info("  %s = %s", key, value)
        with open(args.mrr_result, "a") as f:
            f.write(json.dumps(results) + "\n")

        # save best model
        output_dir, best_mrr = save_best_model(args, results['eval_mrr'], best_mrr, model, results)
        if output_dir is not None:
            torch.save(best_mrr, os.path.join(output_dir, "best.pt"))
            logger.info(f"Saving {best_mrr} checkpoint to {output_dir}/best.pt")


def runtime(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size * max(1, args.n_gpu), num_workers=0)

    num_training_steps = len(train_dataloader) * args.num_train_epochs

    optimizer, scheduler = setup_optimizer_scheduler(args, model, num_training_steps)
    # Initialize GradScaler for FP16
    scaler = GradScaler(enabled=args.fp16)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size  = %d", args.train_batch_size * max(1, args.n_gpu))
    logger.info("  per optimization steps = %d", len(train_dataloader))
    logger.info("  Total optimization steps = %d", num_training_steps)

    model.zero_grad()
    result_time_In = []
    result_time_Non = []
    for idx in range(args.start_idx, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            nl_inputs = batch[0].to(args.device)
            code_inputs = batch[1].to(args.device)
            nl_mask = batch[2].to(args.device) if args.model_class == ModelClass.GRAPH.value else nl_inputs.ne(
                tokenizer.pad_token_id)
            code_mask = code_inputs.ne(tokenizer.pad_token_id)
            ids = batch[3].to(args.device) if args.model_class == ModelClass.GRAPH.value else None

            start_time = time.time()

            with autocast(device_type='cuda', enabled=args.fp16):
                q = model(nl_inputs=nl_inputs, attention_mask=nl_mask, position_ids=ids)
                k = args.strategy.execute(args, idx, model, code_inputs, code_mask, ids)
                loss = contrastive_loss(q, k, args,
                                        args.temperature if args.hidden_state_method == "avg" else 1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            end_time = time.time()
            execution_time = end_time - start_time
            if args.strategy.manager.in_strategy:
                result_time_In.append(execution_time)
            else:
                result_time_Non.append(execution_time)

            if (step + 1) % args.log_interval == 0:
                logger.info(
                    f"epoch {idx} Policy {'In-Policy' if args.strategy.manager.in_strategy else 'Non-Policy'} step {step + 1} time {round(execution_time, 6)}")

        save_last_model(args, model, optimizer, scheduler, idx)
    logger.info(f"In-Policy Avg time {round(sum(result_time_In) / len(result_time_In), 6)}")
    logger.info(f"Non-Policy Avg time {round(sum(result_time_Non) / len(result_time_Non), 6)}")


def evaluate(args, model, tokenizer, data_file):
    """Evaluate model performance and return MRR and Top-K accuracy."""
    nl_dataset = TextDataset(tokenizer, args, data_file)
    nl_dataloader = DataLoader(nl_dataset, sampler=SequentialSampler(nl_dataset),
                               batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=4)

    code_dataset = nl_dataset if args.dataset == "AdvTest" else TextDataset(tokenizer, args, args.codebase_file)
    code_dataloader = DataLoader(code_dataset, sampler=SequentialSampler(code_dataset),
                                 batch_size=args.eval_batch_size * max(1, args.n_gpu), num_workers=4)

    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(nl_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size * max(1, args.n_gpu))

    model.eval()
    nl_vecs, code_vecs = [], []

    with torch.no_grad():
        for batch in nl_dataloader:
            nl_inputs = batch[0].to(args.device)
            mask = batch[2].to(args.device) if args.model_class == ModelClass.GRAPH.value else nl_inputs.ne(
                tokenizer.pad_token_id)
            ids = batch[3].to(args.device) if args.model_class == ModelClass.GRAPH.value else None
            nl_vec = model(nl_inputs=nl_inputs, attention_mask=mask, position_ids=ids)
            nl_vecs.append(nl_vec.cpu().numpy())

        for batch in code_dataloader:
            code_inputs = batch[1].to(args.device)
            mask = batch[2].to(args.device) if args.model_class == ModelClass.GRAPH.value else code_inputs.ne(
                tokenizer.pad_token_id)
            ids = batch[3].to(args.device) if args.model_class == ModelClass.GRAPH.value else None
            code_vec = model(code_inputs=code_inputs, attention_mask=mask, position_ids=ids)
            code_vecs.append(code_vec.cpu().numpy())

    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = [ex.url for ex in nl_dataset.examples]
    code_urls = [ex.url for ex in code_dataset.examples]
    top_k = [1, 5, 10, 50, 100]
    acc_k = [0] * len(top_k)
    ranks = []

    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        for idx in sort_id[:1000]:
            rank += 1
            if code_urls[idx] == url:
                for i, k in enumerate(top_k):
                    if k >= rank:
                        acc_k[i] += 1
                ranks.append(1 / rank)
                break
        else:
            ranks.append(0)
    # pre_k = np.round(np.array(acc_k) / len(nl_dataset), 6)
    return {
        "eval_mrr": round(float(np.mean(ranks)), 6),
        "top_k": top_k,
        "acc_k": (np.round(np.array(acc_k) / len(nl_dataset), 6)).tolist()
    }


def save_model(model, output_dir, pretrained=False):
    """Save the model to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    try:
        if pretrained:
            model_to_save.save_pretrained(output_dir)
        else:
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, SaveModelFileName.STATE_DIC.value))
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def save_best_model(args, cur_mrr, best_mrr, model, results):
    if cur_mrr > best_mrr:
        best_mrr = results['eval_mrr']
        logger.info("  " + "*" * 20)
        logger.info("  Best mrr:%s", best_mrr)
        logger.info("  " + "*" * 20)
        output_dir = os.path.join(args.output_dir, args.model_name, CheckpointType.BEST_MRR.value)
        save_model(model, output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)
    return os.path.join(args.root_output_dir, args.model_name, "parameter"), best_mrr


def save_parameter(args, output_dir, idx):
    os.makedirs(output_dir, exist_ok=True)
    if args.strategy.manager.k_encoder is not None:
        args.strategy.manager.k_encoder.to("cpu")
    torch.save(args.strategy, os.path.join(output_dir, "strategy.pt"))
    if args.global_step != 0:
        torch.save(args.global_step, os.path.join(output_dir, "global_step.pt"))
    with open(f"{output_dir}/idx.txt", "w") as f:
        f.write(str(idx))
    logger.info(f"Saving {idx} checkpoint to {output_dir}/idx.txt")


def save_last_model(args, model, optimizer, scheduler, idx, root_output_dir=None):
    tmp_output_dir = args.output_dir if root_output_dir is None else root_output_dir
    output_dir = os.path.join(tmp_output_dir, args.model_name,
                              CheckpointType.LAST_MRR.value)
    save_model(model, output_dir)
    save_parameter(args, os.path.join(tmp_output_dir, args.model_name, "parameter"), idx)
    logger.info("Saving model checkpoint to %s", output_dir)
    try:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
