'''
    @Project ：
    @File    ：features.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/25 21:43
    @Description :
'''


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 nl_ids,
                 code_ids,
                 url
                 ):
        self.code_ids = code_ids
        self.nl_ids = nl_ids
        self.url = url


class InputFeatures_graph(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 nl_tokens,
                 nl_ids,
                 url
                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
