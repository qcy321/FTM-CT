import logging
import threading


# 创建自定义格式化器
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',  # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',  # 红色
        'CRITICAL': '\033[91m'  # 红色
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        levelname = record.levelname
        color = self.COLORS.get(levelname, self.RESET_COLOR)
        thread_name = threading.current_thread().name
        formatted_message = super().format(record)
        return f"{color}{thread_name} - {formatted_message}{self.RESET_COLOR}"


class Log:
    def __init__(self, level=logging.DEBUG):
        self.logger = logging.getLogger()
        logger = logging.getLogger()
        logger.setLevel(level)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 创建自定义格式化器并添加到处理器
        formatter = ColoredFormatter(fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
