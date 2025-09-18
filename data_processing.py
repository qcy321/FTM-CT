import argparse
import json
import multiprocessing
import os.path
import logging
import tokenize
from io import StringIO

import pandas as pd

from util import Log, run, FunctionInf, parse_func_code, split_task, DfData

logger = logging.getLogger(__name__)


def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    temp = []
    for x in out.split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def single_process(df: pd.DataFrame) -> pd.DataFrame:
    if "code" not in df.columns:
        # Modify a single column name
        df.rename(columns={'function': 'code', 'function_tokens': 'code_tokens'}, inplace=True)
    return df


def mult_data_processing(args, df_data: DfData, chunk_size: int = 500):
    tasks: list[FunctionInf] = []
    df_list: list[pd.DataFrame] = split_task(df_data.df, chunk_size)
    for df in df_list:
        tasks.append(FunctionInf(single_process, (df,)))
    new_list: list[pd.DataFrame] = run(args.num_processes, tasks, df_data.file)
    logger.info("-----合并数据-----")
    all_df = pd.concat(new_list, ignore_index=True)
    logger.info("-----存储数据-----")
    all_df.to_json(f"{args.data_dir[args.dataset]}/{df_data.file}", orient='records',
                   indent=None if df_data.file.endswith(".jsonl") else 4,
                   lines=True if df_data.file.endswith(".jsonl") else False)
    logger.info("-----存储完成-----")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_processes", type=int, default=16,
                        help='执行的进程个数')
    parser.add_argument("--dataset", default="CosQA", type=str,
                        help='Choose one from AdvTest、CosQA')

    args = parser.parse_args()
    args.data_dir = {
        "AdvTest": "./dataset/AdvTest/python",
        "CosQA": "./dataset/CosQA/python",
    }
    files = {
        "AdvTest": ["train.jsonl", "test.jsonl", "valid.jsonl"],
        "CosQA": ["code_idx_map.txt", "cosqa-retrieval-train-19604.json", "cosqa-retrieval-test-500.json",
                  "cosqa-retrieval-dev-500.json"],
    }
    for file in files[args.dataset]:
        if not os.path.exists(args.data_dir[args.dataset] + "/" + file):
            logger.warning(f"{args.data_dir[args.dataset]}/{file}，不存在")
            continue
        if "code_idx_map" in file:
            new_df = pd.DataFrame(
                columns=["idx", "doc", "code", "code_tokens", "docstring_tokens", "label", "retrieval_idx"])
            with open(args.data_dir[args.dataset] + "/" + file) as f:
                js = json.load(f)
                for key in js:
                    pa = parse_func_code(key)
                    if pa is not None:
                        le = new_df.shape[0]
                        new_df.loc[le] = {"idx": "", "doc": "", "code": pa.source_code,
                                          "code_tokens": remove_comments_and_docstrings(pa.source_code).split(),
                                          "docstring_tokens": "", "label": "",
                                          "retrieval_idx": js[key]}
            mult_data_processing(args, DfData(new_df, os.path.splitext(os.path.basename(file))[0] + ".json"))
        else:
            mult_data_processing(args, DfData(
                pd.read_json(args.data_dir[args.dataset] + "/" + file,
                             lines=True if file.endswith(".jsonl") else False),
                file))


if __name__ == '__main__':
    Log(logging.INFO)
    multiprocessing.freeze_support()
    main()
