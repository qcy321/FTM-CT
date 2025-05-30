import logging
import multiprocessing

from tqdm import tqdm
from typing import TypeVar, Generic

T = TypeVar('T')  # 声明一个泛型类型 T
logger = logging.getLogger(__name__)


class FunctionInf(Generic[T]):

    def __init__(self, func: callable, args: tuple) -> None:
        self.func = func
        self.args = args

    def run(self) -> T:
        return self.func(*self.args)


def process_task(func_inf: FunctionInf):
    """
    要执行的任务
    :return: 任务返回值
    """
    # 任务执行
    try:
        result = func_inf.run()
        return result
    except Exception as e:
        logger.error(e)
        logger.warning("该任务没有返回值")
        return None


def worker(func_inf: FunctionInf):
    """
    工作进程函数，执行单个任务
    """
    result = process_task(func_inf)
    return result


def split_task(data: T, chunk_size: int = 10000) -> list[T]:
    """
    数据划分
    :param data: 原数据
    :param chunk_size: 划分大小
    :return:
    """
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def run(num_processes: int, tasks: list[FunctionInf], task_name: str = "task") -> list[T]:
    # 创建一个进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []

        logger.info("-----多进程成功启动-----")

        # 使用 tqdm 创建进度条并实时更新
        with tqdm(total=len(tasks), desc=f'Progress - {task_name}', ncols=80) as pbar:
            # 使用 imap_unordered 逐步获取结果，并更新进度条
            for result in pool.imap_unordered(worker, tasks):
                results.append(result)
                pbar.update(1)

        logger.info("-----任务全部执行完毕-----")
        return results
