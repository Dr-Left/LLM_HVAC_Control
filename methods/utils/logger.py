import logging
import os
from datetime import datetime

import colorlog

# 创建logs目录
os.makedirs("logs", exist_ok=True)

# 设置日志
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Global log filename for all modules
LOG_FILENAME = f"logs/llm_{timestamp}.log"


# 定义ANSI颜色代码
class ColorCodes:
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD_RED = "\033[1;31m"


# 创建一个支持颜色的文件格式化器
class ColoredFileFormatter(logging.Formatter):
    def format(self, record):
        color_code = {
            "DEBUG": ColorCodes.CYAN,
            "INFO": ColorCodes.GREEN,
            "WARNING": ColorCodes.YELLOW,
            "ERROR": ColorCodes.RED,
            "CRITICAL": ColorCodes.BOLD_RED,
        }.get(record.levelname, ColorCodes.WHITE)

        record.msg = f"{color_code}{record.msg}{ColorCodes.RESET}"
        return super().format(record)


# Keep track of handlers to avoid adding multiple handlers to the same file
_file_handler = None


def setup_logger(name, log_file_prefix=""):
    global _file_handler

    # Configure the logger
    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it's already configured
    if logger.handlers:
        return logger

    # Set the overall logger level
    logger.setLevel(logging.DEBUG)

    # Create file handler if it doesn't exist yet
    if _file_handler is None:
        _file_handler = logging.FileHandler(LOG_FILENAME)
        _file_handler.setLevel(logging.DEBUG)
        # Include module name in file logs to maintain context
        file_formatter = ColoredFileFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        _file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(_file_handler)

    # Create terminal handler (each logger gets its own console handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
