'''
    @Project ：
    @File    ：utils.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/25 21:38
    @Description :
'''
import copy
import logging
import os
import random
import shutil

import numpy as np
import torch

from typing import Optional, Any

from torch.nn import CrossEntropyLoss
from common import CheckpointType, SaveModelFileName

from parsers import (remove_comments_and_docstrings,
                     tree_to_token_index,
                     index_to_code_token,
                     tree_to_variable_index)

logger = logging.getLogger(__name__)


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
        # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


def contrastive_loss(q, k, args, temperature=0.05):
    batch_size = q.shape[0]

    # similarity
    sim = torch.einsum('ac,bc->ab', [q, k])

    loss = CrossEntropyLoss()(sim / temperature, torch.arange(batch_size, device=args.device))
    return loss


def update_momentum_encoder(q_encoder, k_encoder, m=0.999):
    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


class KEncoderManager:
    def __init__(self):
        self.k_encoder: Optional[Any] = None  # Cached model instances
        self.in_strategy: bool = False  # Is it in strategy mode?
        self.device = None
        self.count: int = 0  # Current time step (used for increase/decrease strategy)
        self.interval: int = 0  # Interval (different strategies have different meanings)
        self.epoch = 0

    def reset(self):
        """Reset status"""
        self.k_encoder = None
        self.in_strategy = False

    def get_encoder(self, args, model) -> Any:
        """Load and return the cached model"""
        if args.finetune_checkpoint == "best":
            output_dir = os.path.join(args.output_dir, args.model_name,
                                      CheckpointType.BEST_MRR.value,
                                      SaveModelFileName.STATE_DIC.value)
        else:
            output_dir = os.path.join(args.root_output_dir, args.model_name,
                                      CheckpointType.LAST_MRR.value,
                                      SaveModelFileName.STATE_DIC.value)
        # Initialize self.k_encoder to avoid undefined
        self.k_encoder = None
        try:
            self.k_encoder = copy.deepcopy(model)
            self.k_encoder.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
            self.device = args.device
        except Exception as e:
            # Output error log
            logger.error(
                f"Failed: {str(e)}. This might be due to the {output_dir} folder not existing.")

            # Get the checkpoint directory
            checkpoint_dir = os.path.join(args.output_dir, args.model_name, CheckpointType.LAST_MRR.value)

            # Delete the checkpoint directory (if it exists)
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
                except Exception as delete_error:
                    logger.error(f"Failed to delete checkpoint directory {checkpoint_dir}: {str(delete_error)}")

            # Throw an exception to avoid returning invalid objects
            raise RuntimeError(
                f"Model loading failed. Please check the checkpoint at {output_dir} and re-run after fixing the issue.")

            # Verify that self.k_encoder is valid
        if self.k_encoder is None:
            raise ValueError("k_encoder was not initialized properly.")

        return self.k_encoder

    def forward(self, model, code_inputs, attention_mask, position_ids) -> Any:
        """Use the current model for inference"""
        return model(code_inputs=code_inputs, attention_mask=attention_mask, position_ids=position_ids)

    def forward_with_k_encoder(self, code_inputs, attention_mask, position_ids) -> Any:
        """Use cached model for inference (no gradients)"""
        with torch.no_grad():
            if next(self.k_encoder.parameters()).device != self.device:
                self.k_encoder.to(self.device)
            return self.k_encoder(code_inputs=code_inputs, attention_mask=attention_mask, position_ids=position_ids)


def process_in_chunks(code_inputs, mask, ids, model, chunk_size=256):
    with torch.no_grad():
        total_size = code_inputs.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        results = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)

            inputs_chunk = code_inputs[start_idx:end_idx]
            mask_chunk = mask[start_idx:end_idx] if mask is not None else None
            ids_chunk = ids[start_idx:end_idx] if ids is not None else None

            result = model(inputs_chunk, mask_chunk, ids_chunk)
            if result is not None:
                results.append(result)

        return torch.cat(results, dim=0)


class CacheQueue:
    def __init__(self, queue_size, device, fp16=False):
        self.device = device
        self.fp16 = fp16
        self.queue_size = queue_size
        self.dtype = torch.float16 if fp16 else torch.float32
        self.reset()

    def reset(self):
        # Initialize the negative sample queue and dynamically set FP16 or FP32
        self.queue = [[]] * self.queue_size
        self.real_size = 0
        self.queue_ptr = 0

    def get_queue(self):
        return torch.stack(self.queue[:self.real_size], dim=0).to(device=self.device, dtype=self.dtype)

    def update(self, k):
        batch_size = k.shape[0]
        ptr = self.queue_ptr % self.queue_size
        k_detached = k.detach().to(device="cpu", dtype=self.dtype, copy=True)

        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr + batch_size] = k_detached
        else:
            end_size = self.queue_size - ptr
            self.queue[ptr:] = k_detached[:end_size]
            self.queue[:batch_size - end_size] = k_detached[end_size:]

        self.queue_ptr += batch_size
        self.real_size = min(self.queue_size, self.real_size + batch_size)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
