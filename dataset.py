'''
    @Project ：
    @File    ：dataset.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/26 19:50
    @Description :
'''
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from converter import FeatureConverterRegistry, FeatureConverter


def get_cache_file_path(args, prefix):
    if args.model_class != "base":
        cache_file = f"{args.output_dir}/{args.model_class}_{prefix}_{args.dataset}"
    else:
        cache_file = args.output_dir + '/' + prefix + '_' + args.dataset
    suffix = '.pt'
    return cache_file + "_codetest_" + str(args.test_data_size) + suffix if args.code_testing else cache_file + suffix


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, pool=None):
        self.args = args
        prefix = os.path.splitext(os.path.basename(file_path))[0]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        cache_file = get_cache_file_path(args, prefix)
        if not os.path.exists(cache_file):
            self.examples = []
            data = []
            if args.code_testing:
                df = pd.read_json(file_path, lines=True if file_path.endswith(".jsonl") else False).head(
                    args.test_data_size)
            else:
                df = pd.read_json(file_path, lines=True if file_path.endswith(".jsonl") else False)
            strategy = args.strategy
            args.strategy = None
            for i in range(df.shape[0]):
                # if not df.loc[i]["opcode_string"].strip() == "":
                data.append((df.loc[i], tokenizer, args))
            converter: FeatureConverter = FeatureConverterRegistry.get_converter(args.model_class)
            self.examples = pool.map(converter.convert_examples_to_features, tqdm(data, total=len(data)))
            args.strategy = strategy
            torch.save(self.examples, cache_file)
        if os.path.exists(cache_file):
            self.examples = torch.load(cache_file)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.args.model_class == "graph":
            # calculate graph-guided masked function
            attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                                  self.args.code_length + self.args.data_flow_length), dtype=np.bool_)
            # calculate begin index of node and max length of input
            node_index = sum([i > 1 for i in self.examples[item].position_idx])
            max_length = sum([i != 1 for i in self.examples[item].position_idx])
            # sequence can attend to sequence
            attn_mask[:node_index, :node_index] = True
            # special tokens attend to all tokens
            for idx, i in enumerate(self.examples[item].code_ids):
                if i in [0, 2]:
                    attn_mask[idx, :max_length] = True
            # nodes attend to code tokens that are identified from
            for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
                if a < node_index and b < node_index:
                    attn_mask[idx + node_index, a:b] = True
                    attn_mask[a:b, idx + node_index] = True
            # nodes attend to adjacent nodes
            for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
                for a in nodes:
                    if a + node_index < len(self.examples[item].position_idx):
                        attn_mask[idx + node_index, a + node_index] = True

            return (torch.tensor(self.examples[item].nl_ids),
                    torch.tensor(self.examples[item].code_ids),
                    torch.tensor(attn_mask),
                    torch.tensor(self.examples[item].position_idx))
        else:
            return (torch.tensor(self.examples[item].nl_ids),
                    torch.tensor(self.examples[item].code_ids))
