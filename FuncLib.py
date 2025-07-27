import logging
from logging.handlers import RotatingFileHandler
import wx
from typing import Dict
from colorama import Fore, Back
import os

INFO_COLORS = {
    "GREEN": wx.Colour(0, 153, 0),
    "BLUE": wx.Colour(0, 0, 153),
    "YELLOW": wx.Colour(153, 153, 0),
    "RED": wx.Colour(153, 0, 0),
    "BLACK": wx.Colour(0, 0, 0),
    "WHITE": wx.Colour(255, 255, 255),
    "CYAN": wx.Colour(0, 153, 153),
    "MAGENTA": wx.Colour(153, 0, 153),
    "ORANGE": wx.Colour(255, 165, 0),
    "GREY": wx.Colour(128, 128, 128),
}

LEVEL_COLORS = {
    logging.DEBUG: INFO_COLORS["GREY"],
    logging.INFO: INFO_COLORS["BLACK"],
    logging.WARNING: INFO_COLORS["ORANGE"],
    logging.ERROR: INFO_COLORS["RED"],
    logging.CRITICAL: INFO_COLORS["RED"],
}


class WxTextRedirector:
    def __init__(self, text_ctrl, color=wx.BLACK):
        self.text_ctrl = text_ctrl
        self.color = color  # store the default color

    def write(self, string):
        # if string.strip():

        def append():
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(self.color))
            self.text_ctrl.AppendText(string)
            self.text_ctrl.SetDefaultStyle(
                wx.TextAttr(wx.BLACK)
            )  # reset if needed

        wx.CallAfter(append)

    def flush(self):
        pass  # required for compatibility


class TextCtrlHandler(logging.Handler):

    def __init__(self, text_ctrl):
        super().__init__()
        self.text_ctrl = text_ctrl

    def emit(self, record):
        msg = self.format(record)
        msg_color = LEVEL_COLORS.get(record.levelno, wx.Colour(0, 0, 0))
        if record.levelno == logging.INFO:
            for color in INFO_COLORS:
                if f"[{color}]" in msg:
                    msg_color = INFO_COLORS.get(color, wx.Colour(0, 0, 0))
                    msg = msg.replace(f"[{color}]", "")
                    break

        def append():
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(msg_color))
            self.text_ctrl.AppendText(msg + "\n")
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(wx.BLACK))  # Reset

        wx.CallAfter(append)


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
