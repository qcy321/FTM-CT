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

    # 样本相似度
    sim = torch.einsum('ac,bc->ab', [q, k])

    loss = CrossEntropyLoss()(sim / temperature, torch.arange(batch_size, device=args.device))
    return loss


def update_momentum_encoder(q_encoder, k_encoder, m=0.999):
    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


# 定义策略枚举类
class KEncoderManager:
    def __init__(self):
        self.k_encoder: Optional[Any] = None  # 缓存的模型实例
        self.in_strategy: bool = False  # 是否处于策略模式
        self.device = None
        self.count: int = 0  # 当前时间步（用于increase/decrease策略）
        self.interval: int = 0  # 间隔（不同策略有不同含义）
        self.epoch = 0

    def reset(self):
        """重置状态"""
        self.k_encoder = None
        self.in_strategy = False

    def get_encoder(self, args, model) -> Any:
        """加载并返回缓存模型"""
        if args.finetune_checkpoint == "best":
            output_dir = os.path.join(args.output_dir,
                                      CheckpointType.BEST_MRR.value,
                                      SaveModelFileName.STATE_DIC.value)
        else:
            output_dir = os.path.join(args.root_output_dir,
                                      CheckpointType.LAST_MRR.value,
                                      SaveModelFileName.STATE_DIC.value)
        # 初始化 self.k_encoder，避免未定义
        self.k_encoder = None
        try:
            self.k_encoder = copy.deepcopy(model)
            self.k_encoder.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
            self.device = args.device
        except Exception as e:
            # 输出错误日志
            logger.error(
                f"Failed: {str(e)}. This might be due to the {output_dir} folder not existing.")

            # 获取检查点目录
            checkpoint_dir = os.path.join(args.output_dir, CheckpointType.LAST_MRR.value)

            # 删除检查点目录（如果存在）
            if os.path.exists(checkpoint_dir):
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
                except Exception as delete_error:
                    logger.error(f"Failed to delete checkpoint directory {checkpoint_dir}: {str(delete_error)}")

            # 抛出异常，避免返回无效对象
            raise RuntimeError(
                f"Model loading failed. Please check the checkpoint at {output_dir} and re-run after fixing the issue.")

            # 验证 self.k_encoder 是否有效
        if self.k_encoder is None:
            raise ValueError("k_encoder was not initialized properly.")

        return self.k_encoder

    def forward(self, model, code_inputs, attention_mask, position_ids) -> Any:
        """使用当前模型进行推理"""
        return model(code_inputs=code_inputs, attention_mask=attention_mask, position_ids=position_ids)

    def forward_with_k_encoder(self, code_inputs, attention_mask, position_ids) -> Any:
        """使用缓存模型进行推理（无梯度）"""
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


# #配合使用strategy_full.py
# class CacheQueue:
#     def __init__(self, queue_size, device, fp16=False):
#         self.device = device
#         self.fp16 = fp16
#         self.queue_size = queue_size
#         self.dtype = torch.float16 if fp16 else torch.float32
#         self.reset()
#
#     def reset(self):
#         # 初始化负样本队列，动态设置 FP16 或 FP32
#         self.queue = [[]] * self.queue_size
#         self.neg_queue = [[]] * self.queue_size
#         self.real_size = 0
#         self.queue_ptr = 0
#
#     def convert_neg(self, model):
#         code_inputs, mask, ids = self.get_queue()
#         outputs = process_in_chunks(code_inputs, mask, ids, model, 3)
#         self.neg_queue[:self.real_size] = outputs.detach().to(device="cpu",
#                                                               dtype=self.dtype,
#                                                               copy=True)
#
#     def get_queue(self):
#         # 返回队列中的有效数据，分别提取 nl_inputs, code_inputs, mask, ids 并堆叠
#         valid_queue = self.queue[:self.real_size]
#         if not valid_queue or valid_queue[0] is None:
#             return None
#
#         # 提取每个字段的列表
#         code_inputs_list = [item[0] for item in valid_queue]
#         mask_list = [item[1] for item in valid_queue if item[1] is not None]
#         ids_list = [item[2] for item in valid_queue if item[2] is not None]
#
#         # 使用 torch.cat 转换为张量
#         return (
#             torch.cat(code_inputs_list, dim=0).to(device=self.device),
#             torch.cat(mask_list, dim=0).to(device=self.device) if mask_list else None,
#             torch.cat(ids_list, dim=0).to(device=self.device) if ids_list else None
#         )
#
#     def get_neg_queue(self):
#         return torch.stack(self.neg_queue[:self.real_size], dim=0).to(device=self.device, dtype=self.dtype)
#
#     def update(self, code_inputs, mask, ids, k=None):
#         batch_size = code_inputs.shape[0]
#         ptr = self.queue_ptr % self.queue_size
#
#         # 将 batch 拆分为单条数据
#         batch_split = [
#             (
#                 code_inputs[i:i + 1].to(device="cpu"),  # 单条 code_inputs
#                 mask[i:i + 1].to(device="cpu") if mask is not None else None,  # 单条 mask
#                 ids[i:i + 1].to(device="cpu") if ids is not None else None  # 单条 ids
#             ) for i in range(batch_size)
#         ]
#
#         if ptr + batch_size <= self.queue_size:
#             self.queue[ptr:ptr + batch_size] = batch_split
#         else:
#             end_size = self.queue_size - ptr
#             self.queue[ptr:] = batch_split[:end_size]
#             self.queue[:batch_size - end_size] = batch_split[end_size:]
#
#         if k != None:
#             k_detached = k.detach().to(device="cpu", dtype=self.dtype, copy=True)
#
#             if ptr + batch_size <= self.queue_size:
#                 self.neg_queue[ptr:ptr + batch_size] = k_detached
#             else:
#                 end_size = self.queue_size - ptr
#                 self.neg_queue[ptr:] = k_detached[:end_size]
#                 self.neg_queue[:batch_size - end_size] = k_detached[end_size:]
#
#         self.queue_ptr += batch_size
#         self.real_size = min(self.queue_size, self.real_size + batch_size)


class CacheQueue:
    def __init__(self, queue_size, device, fp16=False):
        self.device = device
        self.fp16 = fp16
        self.queue_size = queue_size
        self.dtype = torch.float16 if fp16 else torch.float32
        self.reset()

    def reset(self):
        # 初始化负样本队列，动态设置 FP16 或 FP32
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
