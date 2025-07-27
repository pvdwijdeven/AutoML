import wx
from AutoML_EDA import AutoML_EDA
from pathlib import Path
from FuncLib import Logger
import logging
import sys
import threading


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

    def __init__(self, text_ctrl):
        super().__init__()
        self.text_ctrl = text_ctrl

    def emit(self, record):
        msg = self.format(record)
        msg_color = self.LEVEL_COLORS.get(record.levelno, wx.Colour(0, 0, 0))
        if record.levelno == logging.INFO:
            for color in self.INFO_COLORS:
                if f"[{color}]" in msg:
                    msg_color = self.INFO_COLORS.get(color, wx.Colour(0, 0, 0))
                    msg = msg.replace(f"[{color}]", "")
                    break

        def append():
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(msg_color))
            self.text_ctrl.AppendText(msg + "\n")
            self.text_ctrl.SetDefaultStyle(wx.TextAttr(wx.BLACK))  # Reset

        wx.CallAfter(append)


class AutoMLFrame(wx.Frame):
    def __init__(self, args):
        super().__init__(
            parent=None, title="Self Training ML", size=wx.Size(800, 600)
        )

        self.SetMinSize(wx.Size(600, 400))
        self.Maximize(True)

        panel = wx.Panel(self)

        # === MAIN OUTER SIZER ===
        outer_sizer = wx.BoxSizer(wx.VERTICAL)

        self.buttons_info = {
            "training": {
                "text": "Open training data (.csv/.xlsx)",
                "function": lambda event: self.on_open_file("training"),
            },
            "test": {
                "text": "Open test data (.csv/.xlsx)",
                "function": lambda event: self.on_open_file("test"),
            },
            "StartEDA": {
                "text": "Start EDA",
                "function": lambda event: self.on_start_eda(),
            },
            "Button4": {
                "text": "Button 4",
                "function": lambda event: self.make_placeholder_handler(4),
            },
            "Button5": {
                "text": "Button 5",
                "function": lambda event: self.make_placeholder_handler(5),
            },
            "Button6": {
                "text": "Button 6",
                "function": lambda event: self.make_placeholder_handler(6),
            },
            "OutputFile": {
                "text": "Select Output csv File",
                "function": lambda event: self.on_select_output(),
            },
        }

        for button_name, button_info in self.buttons_info.items():
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)

            # Label aligned to the left of button
            label = wx.StaticText(panel, label="", size=wx.Size(500, -1))
            label.SetBackgroundColour(wx.Colour(220, 220, 220))
            # Corresponding button
            button = wx.Button(
                panel, label=button_info["text"], size=wx.Size(200, -1)
            )
            row_sizer.Add(button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
            row_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            self.buttons_info[button_name]["label"] = label
            self.buttons_info[button_name]["button"] = button

            func = button_info["function"]

            button.Bind(wx.EVT_BUTTON, func)

            # Add row to main vertical sizer
            outer_sizer.Add(row_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # === Add a stretch spacer and exit button at the bottom ===
        outer_sizer.AddStretchSpacer()
        self.log_ctrl = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.HSCROLL | wx.TE_RICH2,
        )

        self.log_handler = TextCtrlHandler(text_ctrl=self.log_ctrl)
        outer_sizer.Add(self.log_ctrl, 1, wx.EXPAND | wx.ALL, 10)
        exit_button = wx.Button(panel, label="Exit")
        exit_button.Bind(wx.EVT_BUTTON, self.on_exit)
        outer_sizer.Add(exit_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        panel.SetSizer(outer_sizer)
        formatter = logging.Formatter(fmt="%(message)s")
        self.log_handler.setFormatter(fmt=formatter)
        self.logger = Logger(
            level_console=Logger.INFO,
            level_file=Logger.DEBUG,
            filename="log/AutoML.log",
            wx_handler=self.log_handler,
        )
        self.log_handler.setLevel(logging.INFO)
        sys.stdout = WxTextRedirector(self.log_ctrl, wx.Colour(255, 0, 0))
        sys.stderr = WxTextRedirector(self.log_ctrl, wx.Colour(150, 0, 0))

        if args.training_data:
            if Path(args.training_data).is_absolute():
                train_path = Path(args.training_data)
            else:
                train_path = args.project_root / args.training_data
            self.buttons_info["training"]["label"].SetLabel(f"{train_path}")
            self.logger.info("[GREEN]training data from command line")
        if args.test_data:
            if Path(args.test_data).is_absolute():
                test_path = Path(args.test_data)
            else:
                test_path = args.project_root / args.test_data
            self.buttons_info["test"]["label"].SetLabel(f"{test_path}")
            self.logger.info("[GREEN]test data from command line")

        self.Show()

    def on_exit(self, event):
        self.Close()

    def on_open_file(self, kind):
        with wx.FileDialog(
            self,
            f"Open {kind} CSV/XLSX file",
            wildcard="Data files (*.csv;*.xlsx)|*.csv;*.xlsx|CSV files (*.csv)|*.csv|Excel files (*.xlsx)|*.xlsx",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = file_dialog.GetPath()
            self.buttons_info[kind]["label"].SetLabel(f"{path}")

    def on_select_output(self):
        with wx.FileDialog(
            self,
            "Save CSV file",
            wildcard="CSV files (*.csv)|*.csv",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = file_dialog.GetPath()
            if not path.endswith(".xlsx"):
                path += ".xlsx"
            self.buttons_info["OutputFile"]["label"].SetLabel(f"Output: {path}")

    def on_start_eda(self):
        thread = threading.Thread(target=self.actual_eda)
        thread.start()

    def actual_eda(self):
        # Placeholder for EDA functionality
        self.buttons_info["StartEDA"]["label"].SetLabel("EDA started")
        result = AutoML_EDA(
            logger=self.logger,
            file_train=self.buttons_info["training"]["label"].GetLabel(),
            file_test=self.buttons_info["test"]["label"].GetLabel(),
        ).perform_eda()
        self.buttons_info["StartEDA"]["label"].SetLabel(result)

    def make_placeholder_handler(self, index):

        self.buttons_info[f"Button{index}"]["label"].SetLabel(
            f"Button {index} clicked"
        )
