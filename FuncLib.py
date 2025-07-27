import logging
from logging.handlers import RotatingFileHandler

from typing import Dict
from colorama import Fore, Back
import os


class CustomFormatter(logging.Formatter):
    """
    Class required for Logger class
    """

    def __init__(
        self,
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    format_debug: str = (
        "%(levelname)s on line %(lineno)d (%(filename)s): %(message)s"
    )
    format_info: str = "%(message)s"
    format_warning: str = "%(levelname)s: %(message)s"
    format_error: str = "%(levelname)s: %(message)s"
    format_critical: str = (
        "%(levelname)s on line %(lineno)d (%(filename)s): %(message)s"
    )
    FORMATS: Dict[int, str] = {
        logging.DEBUG: Fore.GREEN + format_debug + Fore.RESET,
        logging.INFO: Fore.BLUE + format_info + Fore.RESET,
        logging.WARNING: Fore.YELLOW + format_warning + Fore.RESET,
        logging.ERROR: Fore.RED + format_error + Fore.RESET,
        logging.CRITICAL: Fore.BLACK
        + Back.YELLOW
        + format_critical
        + Fore.RESET
        + Back.RESET,
    }

    def format(self, record) -> str:
        log_fmt: str | None = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(fmt=log_fmt)
        return formatter.format(record=record)


class Logger(logging.getLoggerClass()):
    """
    This class sets up a logger with file and console handlers.

    This class sets up a logger with a rotating file handler and a console
    handler.
    The file handler writes log messages to a file, and the console handler
    writes log messages to the console using colors.

    Args:
        level_file (int, optional): The log level for the file handler.
            Defaults to logging.DEBUG.
        level_console (int, optional): The log level for the console handler.
            Defaults to logging.INFO.
        filename (str, optional): The name of the log file.
            Defaults to "my_log.log".

    Returns:
        Logger: a customized logger.
    """

    DEBUG: int = logging.DEBUG
    INFO: int = logging.INFO
    WARNING: int = logging.WARNING
    ERROR: int = logging.ERROR
    CRITICAL: int = logging.CRITICAL

    def __init__(
        self,
        level_file: int = logging.DEBUG,
        level_console: int = logging.INFO,
        filename: str = "my_log.log",
        wx_handler=None,
    ) -> None:
        super().__init__(name="my_logger", level=logging.DEBUG)
        self.setup_logger(
            level_file=level_file,
            level_console=level_console,
            filename=filename,
            wx_handler=wx_handler,
        )

    def setup_logger(
        self,
        level_file: int = logging.DEBUG,
        level_console: int = logging.INFO,
        filename: str = "my_log.log",
        wx_handler=None,
    ) -> None:
        self.setLevel(level=level_file)
        if filename != "":
            log_folder = os.path.dirname(filename)
            if log_folder and not os.path.exists(log_folder):
                os.makedirs(log_folder)
            log_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(filename)-20s | line %(lineno)-4d | %(message)s ",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            max_log_size = 5 * 1024 * 1024
            backup_count = 3
            file_handler = RotatingFileHandler(
                filename=filename,
                maxBytes=max_log_size,
                backupCount=backup_count,
            )
            file_handler.setFormatter(fmt=log_formatter)
            self.addHandler(hdlr=file_handler)

        # Define a Handler for console output
        console = logging.StreamHandler(stream=None)
        console.setFormatter(fmt=CustomFormatter())
        self.addHandler(hdlr=console)
        if wx_handler is None:
            console.setLevel(level=level_console)
        else:
            self.addHandler(hdlr=wx_handler)
            console.setLevel(level=logging.CRITICAL)
