'''
    @Project ：
    @File    ：common.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/26 19:42
    @Description :
'''
from enum import Enum
from typing import Dict, List, Tuple
# from parsers import DFG_python
from parsers import DFG_python, DFG_java, DFG_ruby, DFG_go, DFG_php, DFG_javascript
from tree_sitter import Language, Parser

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'ruby': DFG_ruby,
    'go': DFG_go,
    'php': DFG_php,
    'javascript': DFG_javascript
}
# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parsers/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


class FinetuneStrategy(Enum):
    CROSS = "cross"
    INCREASE = "increase"
    DECREASE = "decrease"
    DIVIDE = "divide"
    NONE = "none"


class ModelClass(Enum):
    BASE = "base"
    UNIX = "unix"
    T5 = "T5"
    BART = "Bart"
    SODA = "soda"
    GRAPH = "graph"
    QWEN = "qwen"


class SaveModelFileName(Enum):
    STATE_DIC = "model.bin"
    SAFETENSORS = "model.safetensors"


MODEL_CLASS_MAPPING = {
    "roberta-base": ModelClass.BASE.value,
    "microsoft/codebert-base": ModelClass.BASE.value,
    "microsoft/unixcoder-base": ModelClass.UNIX.value,
    "Salesforce/codet5-base": ModelClass.T5.value,
    "uclanlp/plbart-base": ModelClass.BART.value,
    "microsoft/graphcodebert-base": ModelClass.GRAPH.value,
    "DeepSoftwareAnalytics/CoCoSoDa": ModelClass.UNIX.value,
    "BAAI/bge-code-v1": ModelClass.QWEN.value,
    "Qwen/Qwen3-Embedding-0.6B": ModelClass.QWEN.value
}


class CheckpointType(Enum):
    LAST_MRR = "checkpoint-last-mrr"
    BEST_MRR = "checkpoint-best-mrr"
    MAX_MRR = "checkpoint-max-mrr"
    PER = "checkpoint-per"
    FINAL = "checkpoint-final"


# Dataset path and file mapping
DATA_CONFIG: Dict[str, Tuple[str, List[str]]] = {
    "CSN": ("./dataset/CSN",
            ["codebase.jsonl", "train.jsonl", "test.jsonl", "valid.jsonl"]),
    "AdvTest": ("./dataset/AdvTest",
                ["valid.jsonl", "train.jsonl", "test.jsonl", "valid.jsonl"]),
    "CosQA": ("./dataset/CosQA",
              ["code_idx_map.json", "cosqa-retrieval-train-19604.json",
               "cosqa-retrieval-test-500.json", "cosqa-retrieval-dev-500.json"]),
    "GPD": ("./dataset/GPD",
            ["codebase.jsonl", "train.jsonl", "test.jsonl", "valid.jsonl"])
}

# Mapping of file types to indexes
FILE_TYPE_MAPPING = {
    "codebase_file": 0,
    "train_data_file": 1,
    "test_data_file": 2,
    "eval_data_file": 3
}
