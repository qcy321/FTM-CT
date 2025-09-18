'''
    @Project ：
    @File    ：interface.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/3/26 17:02
    @Description :
'''


# 基础策略接口
class Strategy:
    def __init__(self, manager, cache):
        self.manager = manager
        self.cache = cache

    def set_interval(self, epoch):
        raise NotImplementedError("Subclasses must implement the set_interval method")

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        """Execute the strategy for processing code inputs and masks.

                This method defines how the strategy processes the input data based on the current epoch and arguments.
                Subclasses must implement this method to provide specific logic for forwarding the model or encoder.

                Args:
                    args (object): Configuration object containing strategy parameters such as interval and finetune_strategy.
                    epoch (int): The current training epoch, used to determine the strategy behavior.
                    model (object): The model instance to be used for forward processing.
                    code_inputs (torch.Tensor): Tensor containing tokenized code input data.
                    mask (torch.Tensor): Attention mask tensor corresponding to code_inputs.
                    ids (torch.Tensor): Position IDs tensor corresponding to code_inputs.

                Returns:
                    torch.Tensor: The output tensor resulting from the forward pass of the model or encoder.

                Raises:
                    NotImplementedError: If the subclass does not implement this method.
                """
        raise NotImplementedError("Subclasses must implement the execute method")


class FeatureConverter:

    def convert_all(self, items):
        datas, tokenizer, args = items
        result = []
        for js in datas:
            out = self.convert_examples_to_features((js, tokenizer, args))
            result.append(out)
        return result

    def convert_examples_to_features(self, item):
        """Convert examples to features.

                Args:
                    item (tuple): A tuple containing the following elements:
                        - js (dict): Input data containing 'code_tokens', 'docstring_tokens', etc.
                        - tokenizer: Tokenizer for tokenizing and converting to IDs.
                        - args: Configuration parameters including code_length and nl_length.

                Returns:
                    InputFeatures: Converted feature object.

                Raises:
                    NotImplementedError: If the subclass does not implement this method.
                """
        raise NotImplementedError("Subclasses must implement the convert_examples_to_features method")
