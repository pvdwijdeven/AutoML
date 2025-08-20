import wx
from eda import AutoML_EDA
from modelling import AutomlModeling
from pathlib import Path
from library import Logger, TextCtrlHandler, WxTextRedirector
import logging
import sys
import threading
import os


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

        # Directory where this current Python file lives (i.e., automl/gui/)
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Go one level up to automl/
        automl_dir = os.path.dirname(current_dir)

        # Construct full path to the icon file inside automl/data/icons/
        icon_path = os.path.join(automl_dir, "data", "icons", "automl_icon.png")

        icon = wx.Icon(icon_path, wx.BITMAP_TYPE_PNG)
        self.SetIcon(icon)

        self.buttons_info = {
            "training": {
                "text": "Open training data (.csv/.xlsx)",
                "function": lambda event: self.on_open_file("training"),
            },
            "target": {
                "text": "target column",
                "function": lambda event: self.on_get_target(),
            },
            "scoring": {
                "text": "scoring method",
                "function": lambda event: self.on_get_scoring(),
            },
            "description": {
                "text": "Select description file",
                "function": lambda event: self.on_open_file("description"),
            },
            "test": {
                "text": "Open test data (.csv/.xlsx)",
                "function": lambda event: self.on_open_file("test"),
            },
            "update_script": {
                "text": "Open update script (.py)",
                "function": lambda event: self.on_open_file("update_script"),
            },
            "ReportFile": {
                "text": "Select report file (.html)",
                "function": lambda event: self.on_select_output("ReportFile"),
            },
            "StartEDA": {
                "text": "Start EDA",
                "function": lambda event: self.on_start_eda(),
            },
            # "Startprepro": {
            #     "text": "Start preprocessing",
            #     "function": lambda event: self.on_start_prepro(),
            # },
            "title": {
                "text": "Title for EDA Report",
                "function": lambda event: self.on_get_title(),
            },
            "OutputFile": {
                "text": "Select Output html File",
                "function": lambda event: self.on_select_output("OutputFile"),
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
            filename=args.logger_file,
            wx_handler=(
                self.log_handler
                if (not args.nogui) or (args.nogui and args.silent)
                else None
            ),
        )
        self.log_handler.setLevel(logging.INFO)
        if not args.nogui:
            sys.stdout = WxTextRedirector(self.log_ctrl, wx.Colour(255, 0, 0))
            sys.stderr = WxTextRedirector(self.log_ctrl, wx.Colour(150, 0, 0))

        if args.training_data:
            if Path(args.training_data).is_absolute():
                train_path = Path(args.training_data)
            else:
                train_path = args.project_root / args.training_data
            self.buttons_info["training"]["label"].SetLabel(f"{train_path}")
            self.logger.debug("[GREEN]training data from command line")
        if args.update_script:
            if Path(args.update_script).is_absolute():
                update_script_path = Path(args.update_script)
            else:
                update_script_path = args.project_root / args.update_script
            self.buttons_info["update_script"]["label"].SetLabel(
                f"{update_script_path}"
            )
            self.logger.debug("[GREEN]test data from command line")
        if args.test_data:
            if Path(args.test_data).is_absolute():
                test_path = Path(args.test_data)
            else:
                test_path = args.project_root / args.test_data
            self.buttons_info["test"]["label"].SetLabel(f"{test_path}")
            self.logger.debug("[GREEN]test data from command line")
        if args.description:
            if Path(args.description).is_absolute():
                test_path = Path(args.description)
            else:
                test_path = args.project_root / args.description
            self.buttons_info["description"]["label"].SetLabel(f"{test_path}")
            self.logger.debug("[GREEN]column description from command line")
        if args.report_file:
            if Path(args.report_file).is_absolute():
                report_path = Path(args.report_file)
            else:
                report_path = args.project_root / args.report_file
            self.buttons_info["ReportFile"]["label"].SetLabel(f"{report_path}")
            self.logger.debug("[GREEN]report file from command line")
        if args.output_file:
            if Path(args.output_file).is_absolute():
                output_path = Path(args.output_file)
            else:
                output_path = args.project_root / args.output_file
            self.buttons_info["OutputFile"]["label"].SetLabel(f"{output_path}")
            self.logger.debug("[GREEN]output file from command line")
        if args.title:
            title = args.title
            self.buttons_info["title"]["label"].SetLabel(f"{title}")
            self.logger.debug("[GREEN]title from command line")
        if args.target:
            target = args.target
            self.buttons_info["target"]["label"].SetLabel(f"{target}")
            self.logger.debug("[GREEN]target from command line")
        if args.scoring:
            scoring = args.scoring
            self.buttons_info["scoring"]["label"].SetLabel(f"{scoring}")
            self.logger.debug("[GREEN]scoring from command line")
        if args.nogui:
            self.nogui = True
            self.logger.info("[GREEN]CLI mode, GUI will not be shown")

        else:
            self.nogui = False
            self.Show()
        if args.EDA:
            self.on_start_eda()
        # if args.prepro:
        #     self.on_start_prepro()

    def on_exit(self, event):
        self.Close()

    def on_get_title(self):
        dlg = wx.TextEntryDialog(
            self, "Enter title for EDA Report:", "EDA Report Title"
        )
        if dlg.ShowModal() == wx.ID_OK:
            title = dlg.GetValue()
            self.buttons_info["title"]["label"].SetLabel(title)
        dlg.Destroy()

    def on_get_target(self):
        dlg = wx.TextEntryDialog(
            self, "Enter target column for training data:", "Target column"
        )
        if dlg.ShowModal() == wx.ID_OK:
            title = dlg.GetValue()
            self.buttons_info["target"]["label"].SetLabel(title)
        dlg.Destroy()

    def on_get_scoring(self):
        dlg = wx.TextEntryDialog(
            self, "Enter scoring method:", "Scoring method"
        )
        if dlg.ShowModal() == wx.ID_OK:
            title = dlg.GetValue()
            self.buttons_info["scoring"]["label"].SetLabel(title)
        dlg.Destroy()

    def on_open_file(self, kind: str):
        if kind == "description":
            title = "column description file"
            wildcard = "description files (*.txt)|*.txt"
        elif kind == "update_script":
            title = "Update script file"
            wildcard = "Python files (*.py)|*.py"
        else:
            wildcard = "Data files (*.csv;*.xlsx)|*.csv;*.xlsx|CSV files (*.csv)|*.csv|Excel files (*.xlsx)|*.xlsx"
            title = f"Open {kind} CSV/XLSX file"
        with wx.FileDialog(
            self,
            title,
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = file_dialog.GetPath()
            self.buttons_info[kind]["label"].SetLabel(f"{path}")

    def on_select_output(self, kind: str):
        with wx.FileDialog(
            self,
            f"Select {kind}",
            wildcard=(
                "CSV files (*.csv)|*.csv"
                if kind == "csv_file"
                else "HTML files (*.html)|*.html"
            ),
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = file_dialog.GetPath()
            self.buttons_info[kind]["label"].SetLabel(f"{path}")

    def on_start_eda(self):
        thread = threading.Thread(target=self.actual_eda)
        thread.start()

    def actual_eda(self):
        # Placeholder for EDA functionality
        self.buttons_info["StartEDA"]["label"].SetLabel("EDA started")
        current_EDA = AutoML_EDA(
            logger=self.logger,
            report_file=self.buttons_info["ReportFile"]["label"].GetLabel(),
            file_train=self.buttons_info["training"]["label"].GetLabel(),
            file_test=self.buttons_info["test"]["label"].GetLabel(),
            title=self.buttons_info["title"]["label"].GetLabel(),
            target=self.buttons_info["target"]["label"].GetLabel(),
            scoring=self.buttons_info["scoring"]["label"].GetLabel(),
            description=self.buttons_info["description"]["label"].GetLabel(),
            nogui=self.nogui,
            update_script=self.buttons_info["update_script"][
                "label"
            ].GetLabel(),
        )
        result = current_EDA.perform_eda()

        self.buttons_info["StartEDA"]["label"].SetLabel(result)
        assert current_EDA.df_train is not None
        self.mymodels = AutomlModeling(
            target=current_EDA.target,
            scoring=self.buttons_info["scoring"]["label"].GetLabel(),
            X_original=current_EDA.df_train,
            df_test=current_EDA.df_test,
            output_file=self.buttons_info["OutputFile"]["label"].GetLabel(),
            title=self.buttons_info["title"]["label"].GetLabel(),
            logger=self.logger,
        )
        if self.nogui:
            self.on_exit(event=None)
