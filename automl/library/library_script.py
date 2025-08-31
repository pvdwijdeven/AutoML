# Standard library imports
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def todo() -> None:
    """TODO: implement this function"""
    raise NotImplementedError("Function not yet implemented.")


class SameLineStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            if msg.startswith("\033[2K\r"):  # [FLUSH]
                self.stream.write(msg + "\n")
                self.flush()

            elif msg.startswith("\r"):  # [SAMELINE]
                self.stream.write(msg)
                self.flush()

            else:
                # Ensure previous sameline/flush does not bleed into this
                self.stream.write(msg + self.terminator)
                self.flush()

        except Exception:
            self.handleError(record)


class CustomFormatter(logging.Formatter):
    COLOR_TAGS = {
        "BLACK": "\033[30m",
        "RED": "\033[31m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[33m",
        "BLUE": "\033[34m",
        "MAGENTA": "\033[35m",
        "CYAN": "\033[36m",
        "WHITE": "\033[97m",
        "GREY": "\033[90m",
        "ORANGE": "\033[38;5;208m",
    }

    RESET = "\033[0m"

    FORMATS = {
        logging.DEBUG: "\033[92m"
        + "%(levelname)s on line %(lineno)d (%(filename)s): %(message)s"
        + RESET,
        logging.INFO: "%(message)s",  # color will come from [TAG]
        logging.WARNING: "\033[93m" + "%(levelname)s: %(message)s" + RESET,
        logging.ERROR: "\033[91m" + "%(levelname)s: %(message)s" + RESET,
        logging.CRITICAL: "\033[30m\033[103m"
        + "%(levelname)s on line %(lineno)d (%(filename)s): %(message)s"
        + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        sameline = False
        flush = False

        if isinstance(record.msg, str):
            if "[SAMELINE]" in record.msg:
                sameline = True
                record.msg = record.msg.replace("[SAMELINE]", "")
            if "[FLUSH]" in record.msg:
                flush = True
                record.msg = record.msg.replace("[FLUSH]", "")

            for tag, color_code in self.COLOR_TAGS.items():
                record.msg = record.msg.replace(f"[{tag}]", color_code)

            record.msg += self.RESET

        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt, self.datefmt)
        result = formatter.format(record)

        if flush:
            result = "\033[2K\r" + result
        elif sameline:
            result = "\r" + result

        return result

    # def format(self, record) -> str:
    #     log_fmt: str | None = self.FORMATS.get(record.levelno)
    #     formatter = logging.Formatter(fmt=log_fmt)
    #     return formatter.format(record=record)


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
        filename: Path | str = "my_log.log",
        wx_handler=None,
    ) -> None:
        super().__init__(name="my_logger", level=logging.DEBUG)
        self.setup_logger(
            level_file=level_file,
            level_console=level_console,
            filename=filename,
        )

    def setup_logger(
        self,
        level_file: int = logging.DEBUG,
        level_console: int = logging.INFO,
        filename: Path | str = "my_log.log",
    ) -> None:
        self.setLevel(level=level_file)
        if filename != "":
            self.filename = filename
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
        console = SameLineStreamHandler(stream=sys.stdout)
        console.setFormatter(fmt=CustomFormatter())
        self.addHandler(hdlr=console)
        console.setLevel(level=level_console)
