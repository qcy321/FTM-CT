'''
    @Project ：CodeOpBERT
    @File    ：pre_model.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2024/4/2 10:00
    @Description : Enhanced implementation of model selection and processing for CodeOpBERT with factory pattern
'''
import torch
import torch.nn as nn
from enum import Enum
from typing import Optional, Any, Dict, Callable
import torch.nn.functional as F


# Define enumeration classes for model types and hidden state methods
class HiddenStateMethod(Enum):
    AVG = "avg"
    CLS = "cls"
    LAST = "last"


# Custom exception class
class ModelSelectionError(ValueError):
    """Raised when an invalid model class or hidden state method is specified"""

    def __init__(self, message: str):
        super().__init__(message)


def normalize_output(outputs: torch.Tensor, p: float = 2.0, dim: int = 1) -> torch.Tensor:
    """Normalize tensor output"""
    return F.normalize(outputs, p=p, dim=dim)


# Model Selection Factory
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(model_type: str):
    """Decorator, used to register model classes"""

    def decorator(cls):
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def select_model(encoder: Any, args: Any) -> nn.Module:
    """
    Select and construct models based on parameters, using the factory pattern.

    Args:
        encoder: Pre-trained model encoder
        args: Parameter object, containing model_class and hidden_state_method attributes

    Returns:
        nn.Module: Constructed model instance

    Raises:
        ModelSelectionError: If model_class or hidden_state_method is invalid
    """
    model_class = getattr(args, "model_class", "default")
    hidden_state_method = getattr(args, "hidden_state_method", None)

    # Determine the model type based on model_class or hidden_state_method
    if model_class == "graph":
        model_type = "graph"
    elif hidden_state_method == HiddenStateMethod.AVG.value:
        model_type = HiddenStateMethod.AVG.value
    elif hidden_state_method == HiddenStateMethod.CLS.value:
        model_type = HiddenStateMethod.CLS.value
    elif hidden_state_method == HiddenStateMethod.LAST.value:
        model_type = HiddenStateMethod.LAST.value
    else:
        raise ModelSelectionError(
            f"Invalid model_class '{model_class}' or hidden_state_method '{hidden_state_method}'. "
            f"Supported types: {list(MODEL_REGISTRY.keys())}"
        )

    # Use factory mode to select model
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise ModelSelectionError(
            f"No model registered for type '{model_type}'. Available types: {list(MODEL_REGISTRY.keys())}")

    # Instantiate the model
    model = model_cls(encoder)
    return AllModel(model, args)


@register_model("graph")
class GraphModel(nn.Module):
    """Model based on graph structure"""

    def __init__(self, encoder: Any):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

    def forward(self,
                code_inputs: Optional[torch.Tensor] = None,
                nl_inputs: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if code_inputs is not None:
            nodes_mask = position_ids.eq(0)
            token_mask = position_ids.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attention_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            return self.encoder(inputs_embeds=inputs_embeddings,
                                attention_mask=attention_mask,
                                position_ids=position_ids)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=attention_mask)[1]


@register_model(HiddenStateMethod.AVG.value)
class LastAvgModel(nn.Module):
    """Model using the last layer of average pooling"""

    def __init__(self, encoder: Any):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

    def _compute_avg_pooling(self, outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculate average pooling"""
        return (outputs * mask[:, :, None]).sum(1) / mask.sum(-1)[:, None]

    def forward(self,
                code_inputs: Optional[torch.Tensor] = None,
                nl_inputs: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=attention_mask)[0]
            return self._compute_avg_pooling(outputs, attention_mask)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=attention_mask)[0]
            return self._compute_avg_pooling(outputs, attention_mask)


@register_model(HiddenStateMethod.CLS.value)
class BaseModel(nn.Module):
    """Base model using CLS markup output"""

    def __init__(self, encoder: Any):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

    def forward(self,
                code_inputs: Optional[torch.Tensor] = None,
                nl_inputs: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if code_inputs is not None:
            return self.encoder(code_inputs, attention_mask=attention_mask)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=attention_mask)[1]


@register_model(HiddenStateMethod.LAST.value)
class LastModel(nn.Module):
    """Base model using CLS markup output"""

    def __init__(self, encoder: Any):
        super().__init__()
        self.encoder = encoder
        self.config = encoder.config

    def forward(self,
                code_inputs: Optional[torch.Tensor] = None,
                nl_inputs: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if code_inputs is not None:
            last_hidden_states = self.encoder(code_inputs, attention_mask=attention_mask)[0]
        else:
            last_hidden_states = self.encoder(nl_inputs, attention_mask=attention_mask)[0]

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class AllModel(nn.Module):
    """Encapsulation class for post-processing model output"""

    def __init__(self, model: nn.Module, args: Any):
        super().__init__()
        self.model = model
        self.config = model.config
        self.args = args
        self.hidden_state_method = getattr(args, "hidden_state_method", None)

    def forward(self,
                code_inputs: Optional[torch.Tensor] = None,
                nl_inputs: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = self.model(code_inputs, nl_inputs, attention_mask, position_ids)
        # if self.hidden_state_method == HiddenStateMethod.AVG.value:
        return normalize_output(outputs)
        # return outputs

    def save_pretrained(self, path: str) -> None:
        """Save pre-trained model"""
        self.model.encoder.save_pretrained(path)
