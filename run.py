'''
    @Project ：
    @File    ：cross_train.py
    @IDE     ：PyCharm
    @Author  ：qcy123
    @Date    ：2024/11/8 19:32
    @Description :
'''
import argparse
import logging
import os
import shutil

import torch

from common import CheckpointType, DATA_CONFIG, FILE_TYPE_MAPPING, MODEL_CLASS_MAPPING, FinetuneStrategy, \
    SaveModelFileName
from typing import Any, Dict

from strategy import StrategyRegistry
from pre_model import select_model
from utils import set_seed, KEncoderManager, CacheQueue
from train import pre_training, fine_tuning, evaluate, runtime

from transformers import (AutoModel, AutoTokenizer, RobertaTokenizer, T5Model,
                          RobertaModel)
from peft import LoraConfig, get_peft_model, TaskType

# import transformers
# transformers.utils.hub.HF_OFFLINE = True

logger = logging.getLogger(__name__)


class ModelManager:
    """Manage model loading, saving, and device settings"""

    def __init__(self, model_name_or_path: str, model_class: str, device: torch.device):
        self.model_name_or_path = model_name_or_path
        self.model_class = model_class
        self.device = device
        self.model = None

    def load_model(self, checkpoint_path: str = None, strict: bool = False) -> None:
        """Load the model and move it to the specified device"""
        logger.info(f"Loading model from {self.model_name_or_path if checkpoint_path is None else checkpoint_path}")
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

        if self.model_class in ["QWEN"]:
            # 配置 LoRA
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=16,  # rank
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj"],  # Fine-tuned module (adjusted according to the model)
                lora_dropout=0.1,
            )

            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)

        if self.model_class in ["T5"]:
            self.model = self.model.encoder
        self.model = select_model(self.model, args)  # Assume select_model is a defined function

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=self.device),
                                       strict=strict)
        else:
            logger.warning(f"Checkpoint path {checkpoint_path} does not exist, using pretrained weights")

        self.model.to(self.device)
        logger.info(f"Model moved to device: {self.device}")


def setup_device_and_gpu(args):
    """Set device and GPU parameters"""
    device = torch.device(f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = len(args.device_ids)
    args.device = device
    logger.info("Device: %s, Number of GPUs: %d", device, args.n_gpu)


def load_training_state(args) -> None:
    """Set training status, including breakpoint resume information and optimal MRR"""
    path = os.path.join(args.output_dir, args.model_name, "parameter")
    idx_file = os.path.join(path, "idx.txt")
    best_file = os.path.join(path, "best.pt")
    global_step_file = os.path.join(path, "global_step.pt")

    args.start_dup = 0
    args.idx = 0
    args.start_idx = 0
    args.best_mrr = -1.0 if not hasattr(args, 'all_lang') else [-1] * len(args.all_lang)
    args.global_step = 0

    if os.path.exists(global_step_file):
        args.global_step = torch.load(global_step_file, weights_only=False)
        logger.info(f"Resuming training from {args.global_step} steps")

    if os.path.exists(idx_file):
        with open(idx_file, "r") as f:
            args.idx = int(f.read()) + 1
        logger.info(f"Resuming training from epoch {args.idx}")
    else:
        logger.info("Starting training from scratch")

    if os.path.exists(best_file):
        args.best_mrr = torch.load(best_file, weights_only=False)
        logger.info(f"Best MRR from previous training: {args.best_mrr}")
    else:
        logger.info("No best MRR found, starting with 0.0")


def evaluate_or_test(args, model_manager: ModelManager, tokenizer: Any, data_file: str,
                     mode: str = "eval", checkpoint_type: CheckpointType = CheckpointType.LAST_MRR) -> Dict:
    """Perform evaluation or testing"""
    if not args.do_zero_shot:
        checkpoint_path = os.path.join(args.output_dir, args.model_name, checkpoint_type.value,
                                       SaveModelFileName.STATE_DIC.value)
        model_manager.load_model(checkpoint_path=checkpoint_path, strict=False)
    else:
        logger.info(f"***** zero shot eval *****")

    result = evaluate(args, model_manager.model, tokenizer, data_file)
    logger.info(f"***** {mode.capitalize()} results ({checkpoint_type.value}) *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result


def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)

    # Load training status (breakpoint resume information, best MRR)
    load_training_state(args)

    # Initialize the model manager
    model_manager = ModelManager(args.model_name_or_path, args.model_class, args.device)

    logger.info("Training/evaluation parameters %s", args)

    if args.idx != 0:
        checkpoint_path = os.path.join(args.output_dir, args.model_name, CheckpointType.LAST_MRR.value,
                                       SaveModelFileName.STATE_DIC.value)
        if args.train_mode == "pretrain":
            args.start_idx = args.idx % args.num_train_epochs
            args.start_dup = args.idx // (len(args.all_lang) * args.num_train_epochs)
            logger.info(f"Continue from where we left off last time epoch {args.start_idx}-{args.start_dup}")
        else:
            args.start_idx = args.idx
            logger.info(f"Continuing training from epoch {args.start_idx}")
        model_manager.load_model(checkpoint_path=checkpoint_path, strict=False)
    else:
        model_manager.load_model()  # Start with a pre-trained model

    if args.do_zero_shot:
        evaluate_or_test(args, model_manager, tokenizer, args.test_data_file,
                         mode="eval", checkpoint_type=CheckpointType.LAST_MRR)
        return 0

    # train
    if args.do_train:
        if args.train_mode == "finetune":
            fine_tuning(args, model_manager.model, tokenizer)
        elif args.train_mode == "pretrain":
            pre_training(args, model_manager.model, tokenizer)
        else:
            runtime(args, model_manager.model, tokenizer)

    # Evaluate
    results = {}

    if args.do_eval:
        results.update(evaluate_or_test(args, model_manager, tokenizer, args.eval_data_file,
                                        mode="eval", checkpoint_type=CheckpointType.LAST_MRR))

    # Test (using the best model)
    if args.do_test:
        results.update(evaluate_or_test(args, model_manager, tokenizer, args.test_data_file,
                                        mode="test", checkpoint_type=CheckpointType.BEST_MRR))
        # Test (using last checkpoint)
        results.update(evaluate_or_test(args, model_manager, tokenizer, args.test_data_file,
                                        mode="test", checkpoint_type=CheckpointType.LAST_MRR))

    rmm_files(args)

    return results


def rmm_files(args) -> None:
    """
    Test the code execution result and clean up generated files if testing is enabled.

    Args:
        args: Arguments containing configuration, including code_testing and output_dir.
    """
    if args.code_testing:
        logger.info("Code testing completed successfully. The code runs as expected.")
        print("Removing test-generated files and directories.")
        folder_path = args.root_output_dir
        try:
            shutil.rmtree(folder_path)
            os.remove(args.mrr_result)
            print(f"Successfully deleted folder: {folder_path}")
            print(f"Successfully deleted txt: {args.mrr_result}")
        except FileNotFoundError:
            print(f"Folder does not exist: {folder_path}")
        except PermissionError:
            print(f"Permission denied: Unable to delete {folder_path}")
        except Exception as e:
            print(f"Deletion failed: {str(e)}")

        print(f"Please check the output log at: {args.log}")
    else:
        logger.info("Training is complete, delete the intermediate generated files.")
        folder_path = os.path.join(args.root_output_dir, args.model_name, "parameter")
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully deleted folder: {folder_path}")
        except FileNotFoundError:
            print(f"Folder does not exist: {folder_path}")
        except PermissionError:
            print(f"Permission denied: Unable to delete {folder_path}")
        except Exception as e:
            print(f"Deletion failed: {str(e)}")


def setup_output_dir(args) -> None:
    """
    Set the output directory.

    Args:
        args: Command line argument object, must include a lang attribute.
    """
    # Check if output_dir is empty
    if not hasattr(args, 'output_dir') or not args.output_dir:
        raise ValueError(
            "The 'output_dir' attribute is missing or empty. Please provide a valid output directory path.")

    args.output_dir = os.path.join(args.output_dir, args.lang)

    if args.code_testing:
        args.output_dir = os.path.join(args.output_dir, "CodeTest")
        logger.info("****   ****************************   ****")
        logger.info("****   Code testing mode is enabled   ****")
        logger.info("****   ****************************   ****")
        args.max_steps = 100
        args.log_interval = 2
        args.train_batch_size = 2

    args.root_output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory set to: {args.output_dir}")


def setup_device_ids(args) -> None:
    """Set the device ID."""
    gpu_count = torch.cuda.device_count()
    logger.info(f"Detected {gpu_count} GPUs")

    if getattr(args, "device_ids", None) is not None:
        try:
            device_ids = [int(id) for id in args.device_ids.split(",")]
            # 验证设备 ID 是否有效
            for device_id in device_ids:
                if device_id < 0 or device_id >= gpu_count:
                    raise ValueError(f"Invalid device ID {device_id}. Must be in range [0, {gpu_count - 1}]")
        except ValueError as e:
            logger.error(f"Error parsing device_ids: {e}")
            raise
    else:
        device_ids = list(range(gpu_count))
        logger.info(f"No device_ids specified, using all available GPUs: {device_ids}")

    args.device_ids = device_ids
    logger.info(f"Device IDs set to: {args.device_ids}")


def setup_data_paths(args) -> None:
    """Set the dataset file path. """
    # Check if the dataset supports
    if args.dataset not in DATA_CONFIG:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets: {list(DATA_CONFIG.keys())}")
    data_dir, files = DATA_CONFIG[args.dataset]
    if args.train_mode == "pretrain":
        if args.dataset != "CSN":
            raise "Please use the CSN dataset for pre-training"
        args.all_lang = ["go", "java", "javascript", "php", "python", "ruby"]
        args.lang = "all"
        for file_type, idx in FILE_TYPE_MAPPING.items():
            # Generate a list for each file type
            file_paths = [
                os.path.join(data_dir, os.path.join(lang, files[idx]))
                for lang in args.all_lang
            ]
            # Plural form indicates a multi-language file list
            plural_file_type = file_type + "s"
            setattr(args, plural_file_type, file_paths)
            logger.info(f"Set {plural_file_type}: {file_paths}")
    else:
        # Check if lang is empty
        if not hasattr(args, 'lang') or not args.lang:
            raise ValueError(
                "The 'lang' attribute is missing or empty. Please specify a valid language identifier (e.g., 'python', 'java').")

        # Dynamically set the file path
        for file_type, idx in FILE_TYPE_MAPPING.items():
            file_path = os.path.join(data_dir, os.path.join(args.lang, files[idx]))
            setattr(args, file_type, file_path)
            logger.info(f"Set {file_type}: {file_path}")


def setup_strategy(args):
    # Initialize KEncoderManager
    strategy = os.path.join(args.output_dir, args.model_name, "parameter", "strategy.pt")
    if os.path.exists(strategy):
        args.strategy = torch.load(strategy, weights_only=False)
        logger.info("Loading strategy from %s", strategy)
    else:
        k_manager = KEncoderManager()
        queue = CacheQueue(args.queue_size, args.device, args.fp16)

        # Automatic injection strategy
        strategy_name = getattr(args, "finetune_strategy", FinetuneStrategy.NONE.value)
        logger.info(f"Finetune strategy: {strategy_name}")
        args.strategy = StrategyRegistry.get_strategy(strategy_name, k_manager, queue)
        args.strategy.set_interval(args.num_train_epochs)


def setup_model_class(args):
    try:
        # Try to get the corresponding model_class from the mapping relationship
        args.model_class = MODEL_CLASS_MAPPING[args.model_name_or_path]
        logger.info(f"Model class: {args.model_class}")
    except KeyError:
        # If the corresponding model_name_or_path is not found, an error message will be given
        print(f"Error: The model name '{args.model_name_or_path}' is not supported.")
        # You can choose to throw an exception or perform other processing according to actual needs
        args.model_class = None
    return args


def main_with_args(args):
    """Main function entry, set parameters and start training."""

    # Set the data path
    setup_data_paths(args)

    # Set the output directory
    setup_output_dir(args)

    # Set the device ID
    setup_device_ids(args)

    # Set the model type
    setup_model_class(args)

    # Setup CUDA, GPU
    setup_device_and_gpu(args)

    # Set the training strategy
    setup_strategy(args)

    # Set the seed
    set_seed(args.seed)

    # Start the training process
    main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="Output directory for saving model predictions and checkpoints.")

    parser.add_argument("--dataset", default="CSN", type=str,
                        help="Dataset name, choose from CSN, AdvTest, CosQA or GPD.")

    parser.add_argument("--lang", default="python", type=str,
                        help="Programming language, choose from go, java, javascript, php, python, or ruby.")

    parser.add_argument('--log', type=str, default="./train.log",
                        help="Log file path for recording the training process.")
    parser.add_argument('--mrr_result', type=str, default="./mrr_result.txt",
                        help="Path to save the MRR result file.")

    parser.add_argument("--model_name_or_path", default="DeepSoftwareAnalytics/CoCoSoDa", type=str,
                        help="Model checkpoint path or name for weight initialization.")  # microsoft/codebert-base
    parser.add_argument("--model_name", default="CoCoSoDa", type=str,
                        help="Model name.")

    parser.add_argument("--hidden_state_method", default="avg", type=str,
                        help="Method to calculate the hidden state, e.g., cls, avg.")
    parser.add_argument("--train_mode", default="finetune", type=str,
                        help="Training mode, choose from finetune, pretrain or runtime.")

    parser.add_argument("--finetune_strategy", default="cross", type=str,
                        help="Finetuning strategy, choose from cross, increase, decrease, none or divide.")
    parser.add_argument("--finetune_checkpoint", default="best", type=str,
                        help="Checkpoint to use for finetuning, choose from best or last.")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Maximum sequence length for natural language input after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Maximum sequence length for code input after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Maximum sequence length for data flow input after tokenization.")
    parser.add_argument("--queue_size", default=2048, type=int,
                        help="Size of the cache queue, effective only in cross, increase, decrease, or divide finetuning strategies.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_momentum", action='store_true',
                        help="Whether to start momentum.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to start zero_shot eval.")

    parser.add_argument("--code_testing", action='store_true',
                        help="Simple test code can run normally.")
    parser.add_argument('--test_data_size', type=int, default=10,
                        help="Size of test data to use for training.")
    parser.add_argument('--cpu_core', type=int, default=16,
                        help="cpu core number.")

    parser.add_argument("--fp16", action='store_true',
                        help="Whether to use 16-bit floating-point precision for training.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="Initial learning rate for the Adam optimizer.")
    parser.add_argument("--temperature", default=0.05, type=float,
                        help="Temperature value controlling the randomness of model output.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="The maximum gradient norm of gradient clipping, used to control the gradient size.")
    parser.add_argument("--log_interval", default=100, type=float,
                        help="The interval of steps for logging during training, used to control the frequency of log output.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Maximum gradient norm for gradient clipping.")
    parser.add_argument("--max_steps", default=10000, type=int,
                        help="Maximum number of training steps, only used in pretrain mode.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs.")
    parser.add_argument('--device_ids', type=str, default=None,
                        help="List of GPU device IDs to use; defaults to all available GPUs if not specified.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # print arguments
    args = parser.parse_args()

    # set log
    logging.basicConfig(filename=args.log, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    main_with_args(args)
