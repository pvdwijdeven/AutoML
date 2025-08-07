import logging
from logging.handlers import RotatingFileHandler
import wx
import os
import sys


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


WX_INFO_COLORS = {
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

CONSOLE_INFO_COLORS = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[92m",  # Bright Green
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[97m",  # Bright White
    "GREY": "\033[90m",  # Bright Black (looks like grey)
    "ORANGE": "\033[38;5;208m",  # Approximation using 256-color mode
}

LEVEL_COLORS = {
    logging.DEBUG: WX_INFO_COLORS["GREY"],
    logging.INFO: WX_INFO_COLORS["BLACK"],
    logging.WARNING: WX_INFO_COLORS["ORANGE"],
    logging.ERROR: WX_INFO_COLORS["RED"],
    logging.CRITICAL: WX_INFO_COLORS["RED"],
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


class SameLineStreamHandler(logging.StreamHandler):
    def emit(self, record):
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


def overwrite_last_line(text_ctrl, new_text):
    content = text_ctrl.GetValue()
    lines = content.splitlines()

    if lines:
        lines[-1] = new_text  # Replace last line
    else:
        lines.append(new_text)

    text_ctrl.SetValue("\n".join(lines))
    text_ctrl.ShowPosition(text_ctrl.GetLastPosition())


class TextCtrlHandler(logging.Handler):
    def __init__(self, text_ctrl):
        super().__init__()
        self.text_ctrl = text_ctrl
        self.last_line_start = 0  # Track position of last line start

    def emit(self, record):
        msg = self.format(record)
        sameline = "[SAMELINE]" in msg
        msg = msg.replace("[SAMELINE]", "")

        # Determine message color
        msg_color = LEVEL_COLORS.get(record.levelno, wx.Colour(0, 0, 0))
        if record.levelno == logging.INFO:
            for color in WX_INFO_COLORS:
                if f"[{color}]" in msg:
                    msg_color = WX_INFO_COLORS.get(color, wx.Colour(0, 0, 0))
                    msg = msg.replace(f"[{color}]", "")
                    break

        def update():
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(msg_color))

            if sameline:
                # Overwrite the previous line
                end_pos = self.text_ctrl.GetLastPosition()
                self.text_ctrl.Replace(
                    self.last_line_start, end_pos, msg + "\n"
                )
            else:
                # Remember where this new line starts
                self.last_line_start = self.text_ctrl.GetLastPosition()
                self.text_ctrl.AppendText(msg + "\n")

            self.text_ctrl.ShowPosition(self.text_ctrl.GetLastPosition())

        wx.CallAfter(update)


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
        console = SameLineStreamHandler(stream=sys.stdout)
        console.setFormatter(fmt=CustomFormatter())
        self.addHandler(hdlr=console)

        if wx_handler is None:
            console.setLevel(level=level_console)
        else:
            self.addHandler(hdlr=wx_handler)
            console.setLevel(level=logging.CRITICAL)
