'''
    @Project ：
    @File    ：converter.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/26 18:57
    @Description :
'''
from interface import FeatureConverter
from features import InputFeatures_graph, InputFeatures
from common import ModelClass, parsers
from utils import extract_dataflow


class FeatureConverterRegistry:
    _converter = {}

    @classmethod
    def register(cls, *converter_names):
        def decorator(converter_class):
            for converter_name in converter_names:
                cls._converter[converter_name] = converter_class
            return converter_class

        return decorator

    @classmethod
    def get_converter(cls, converter_name):
        converter_class = cls._converter.get(converter_name)
        if converter_class is None:
            # Provide detailed error message in English with implementation guidance
            error_msg = (
                f"Converter '{converter_name}' not found.\n"
                f"Please implement a class inheriting from FeatureConverter and register it using @FeatureConverterRegistry.register('{converter_name}').\n"
                "Here is an example implementation:\n\n"
                "```python\n"
                f"@FeatureConverterRegistry.register('{converter_name}')\n"
                f"class CustomFeatureConverter(FeatureConverter):\n"
                "    def convert_examples_to_features(self, js, tokenizer, args):\n"
                "        # Process code part\n"
                "        code = ' '.join(js['code_tokens']) if isinstance(js['code_tokens'], list) else ' '.join(js['code_tokens'].split())\n"
                "        code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]\n"
                "        code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]\n"
                "        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)\n"
                "        padding_length = args.code_length - len(code_tokens)\n"
                "        code_ids += [tokenizer.pad_token_id] * padding_length\n"
                "\n"
                "        # Process natural language part\n"
                "        nl = ' '.join(js['docstring_tokens']) if isinstance(js['docstring_tokens'], list) else ' '.join(js.get('doc', '').split())\n"
                "        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]\n"
                "        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]\n"
                "        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)\n"
                "        padding_length = args.nl_length - len(nl_ids)\n"
                "        nl_ids += [tokenizer.pad_token_id] * padding_length\n"
                "\n"
                "        return InputFeatures(nl_ids, code_ids, js.get('url', js.get('retrieval_idx', '')))\n"
                "```\n"
                "Ensure your implementation meets the requirements of the FeatureConverter interface."
            )
            raise ValueError(error_msg)
        return converter_class()


@FeatureConverterRegistry.register(ModelClass.BASE.value, ModelClass.T5.value, ModelClass.BART.value)
class FeatureConverterBase(FeatureConverter):

    def convert_examples_to_features(self, item):
        js, tokenizer, args = item
        # code = remove_comments_and_docstrings(js["original_string"])
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        code_tokens = tokenizer.tokenize(code)[:args.code_length - 2]
        code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_tokens)
        code_ids += [tokenizer.pad_token_id] * padding_length

        # nl
        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(nl_ids, code_ids, js['url'] if "url" in js else js["retrieval_idx"])


@FeatureConverterRegistry.register(ModelClass.UNIX.value, ModelClass.SODA.value)
class FeatureConverterUnix(FeatureConverter):
    def convert_examples_to_features(self, item):
        """convert examples to token ids"""
        js, tokenizer, args = item
        code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
        code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
        code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        padding_length = args.code_length - len(code_ids)
        code_ids += [tokenizer.pad_token_id] * padding_length

        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
        nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures(nl_ids, code_ids, js['url'] if "url" in js else js["retrieval_idx"])


@FeatureConverterRegistry.register(ModelClass.GRAPH.value)
class FeatureConverterGraph(FeatureConverter):
    def convert_examples_to_features(self, item):
        js, tokenizer, args = item
        # code
        parser = parsers[args.lang]
        # extract data flow
        code_tokens, dfg = extract_dataflow(js['code'], parser, args.lang)
        code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                       enumerate(code_tokens)]
        ori2cur_pos = {}
        ori2cur_pos[-1] = (0, 0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
        code_tokens = [y for x in code_tokens for y in x]
        # truncating
        code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
        code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
        code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
        dfg = dfg[:args.code_length + args.data_flow_length - len(code_tokens)]
        code_tokens += [x[0] for x in dfg]
        position_idx += [0 for x in dfg]
        code_ids += [tokenizer.unk_token_id for x in dfg]
        padding_length = args.code_length + args.data_flow_length - len(code_ids)
        position_idx += [tokenizer.pad_token_id] * padding_length
        code_ids += [tokenizer.pad_token_id] * padding_length
        # reindex
        reverse_index = {}
        for idx, x in enumerate(dfg):
            reverse_index[x[1]] = idx
        for idx, x in enumerate(dfg):
            dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
        dfg_to_dfg = [x[-1] for x in dfg]
        dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
        length = len([tokenizer.cls_token])
        dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
        # nl
        nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 2]
        nl_tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token]
        nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
        padding_length = args.nl_length - len(nl_ids)
        nl_ids += [tokenizer.pad_token_id] * padding_length

        return InputFeatures_graph(code_tokens, code_ids, position_idx, dfg_to_code, dfg_to_dfg, nl_tokens, nl_ids,
                                   js['url'] if "url" in js else js["retrieval_idx"])
