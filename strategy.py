'''
    @Project ：
    @File    ：strategy.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2025/4/15 18:13
    @Description :
'''
import math

from interface import Strategy
from common import FinetuneStrategy

import torch


class StrategyRegistry:
    _strategies = {}

    @classmethod
    def register(cls, strategy_name):
        def decorator(strategy_class):
            cls._strategies[strategy_name] = strategy_class
            return strategy_class

        return decorator

    @classmethod
    def get_strategy(cls, strategy_name, manager, cache):
        strategy_class = cls._strategies.get(strategy_name)
        if strategy_class is None:
            raise ValueError(f"未知的微调策略: {strategy_name}")
        return strategy_class(manager, cache)


# 具体策略实现（使用注册器自动注入）
@StrategyRegistry.register(FinetuneStrategy.CROSS.value)
class CrossStrategy(Strategy):
    def set_interval(self, epoch):
        self.manager.interval = 2

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        # interval 2 epoch 10
        if (epoch + 1) % self.manager.interval != 0:
            if self.manager.in_strategy:
                self.manager.reset()
            k = self.manager.forward(model, code_inputs, mask, ids)
            outputs = k
        else:
            if not self.manager.in_strategy:
                self.manager.get_encoder(args, model)
                self.manager.in_strategy = True
            k = self.manager.forward_with_k_encoder(code_inputs, mask, ids)
            outputs = torch.cat([k, self.cache.get_queue()], dim=0)
        self.cache.update(k)
        return outputs


@StrategyRegistry.register(FinetuneStrategy.INCREASE.value)
class IncreaseStrategy(Strategy):

    def set_interval(self, epoch):
        self.manager.interval = 2
        self.manager.count = self.manager.interval

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        # interval 2 epoch 9
        if epoch != self.manager.epoch and self.manager.in_strategy:
            self.manager.count += self.manager.interval
        if (epoch + 1) % self.manager.count != 0:
            if self.manager.in_strategy:
                self.manager.reset()
            k = self.manager.forward(model, code_inputs, mask, ids)
            outputs = k
        else:
            if not self.manager.in_strategy:
                self.manager.get_encoder(args, model)
                self.manager.in_strategy = True
                self.manager.interval += 1
                self.manager.epoch = epoch
            k = self.manager.forward_with_k_encoder(code_inputs, mask, ids)
            outputs = torch.cat([k, self.cache.get_queue()], dim=0)
        self.cache.update(k)
        return outputs


@StrategyRegistry.register(FinetuneStrategy.DECREASE.value)
class DecreaseStrategy(Strategy):

    def set_interval(self, epoch):
        self.manager.interval = math.floor((math.sqrt(8 * epoch + 9) - 1) / 2)
        self.manager.count = self.manager.interval

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        # interval 4 epoch 9
        if epoch != self.manager.epoch and self.manager.in_strategy:
            self.manager.count += self.manager.interval
        if (epoch + 1) % self.manager.count != 0:
            if self.manager.in_strategy:
                self.manager.reset()
            k = self.manager.forward(model, code_inputs, mask, ids)
            outputs = k
        else:
            if not self.manager.in_strategy:
                self.manager.get_encoder(args, model)
                self.manager.in_strategy = True
                self.manager.epoch = epoch
                self.manager.interval = max(2, self.manager.interval - 1)
            k = self.manager.forward_with_k_encoder(code_inputs, mask, ids)
            outputs = torch.cat([k, self.cache.get_queue()], dim=0)
        self.cache.update(k)
        return outputs


@StrategyRegistry.register(FinetuneStrategy.DIVIDE.value)
class DivideStrategy(Strategy):

    def set_interval(self, epoch):
        self.manager.interval = math.ceil(epoch * 0.5)

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        # interval 5 epoch 10
        if epoch < self.manager.interval:
            if self.manager.in_strategy:
                self.manager.reset()
            k = self.manager.forward(model, code_inputs, mask, ids)
            outputs = k
        else:
            if not self.manager.in_strategy:
                self.manager.get_encoder(args, model)
                self.manager.in_strategy = True
            with torch.no_grad():
                k = self.manager.forward_with_k_encoder(code_inputs, mask, ids)
                outputs = torch.cat([k, self.cache.get_queue()], dim=0)
        self.cache.update(k)
        return outputs


@StrategyRegistry.register(FinetuneStrategy.NONE.value)
class NoneStrategy(Strategy):

    def set_interval(self, epoch):
        self.manager.interval = 0

    def execute(self, args, epoch, model, code_inputs, mask, ids):
        return self.manager.forward(model, code_inputs, mask, ids)
