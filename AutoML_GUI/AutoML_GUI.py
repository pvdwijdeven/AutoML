import wx


class AutoMLFrame(wx.Frame):
    def __init__(self):
        super().__init__(
            parent=None, title="Self Training ML", size=wx.Size(800, 600)
        )

        self.SetMinSize(wx.Size(600, 400))
        self.Maximize(True)

        panel = wx.Panel(self)

        # === MAIN OUTER SIZER ===
        outer_sizer = wx.BoxSizer(wx.VERTICAL)

        self.buttons_info = {
            "TrainingData": {
                "text": "Open training data (.csv)",
                "kind": "train",
                "function": lambda event: self.on_open_csv("train"),
            },
            "TestData": {
                "text": "Open test data (.csv)",
                "kind": "test",
                "function": lambda event: self.on_open_csv("test"),
            },
            "StartEDA": {
                "text": "Start EDA",
                "kind": None,
                "function": lambda event: self.on_start_eda(),
            },
            "Button4": {
                "text": "Button 4",
                "kind": None,
                "function": lambda event: self.make_placeholder_handler(4),
            },
            "Button5": {
                "text": "Button 5",
                "kind": None,
                "function": lambda event: self.make_placeholder_handler(5),
            },
            "Button6": {
                "text": "Button 6",
                "kind": None,
                "function": lambda event: self.make_placeholder_handler(6),
            },
            "OutputFile": {
                "text": "Select Output csv File",
                "kind": None,
                "function": lambda event: self.on_select_output(),
            },
        }

        for button_name, button_info in self.buttons_info.items():
            row_sizer = wx.BoxSizer(wx.HORIZONTAL)

            # Label aligned to the left of button
            label = wx.StaticText(
                panel, label=f"Label {button_name}", size=wx.Size(500, -1)
            )
            label.SetBackgroundColour(wx.Colour(220, 220, 220))
            # Corresponding button
            button = wx.Button(
                panel, label=button_info["text"], size=wx.Size(200, -1)
            )
            row_sizer.Add(button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
            row_sizer.Add(label, 0, wx.ALIGN_CENTER_VERTICAL)
            self.buttons_info[button_name]["label"] = label
            self.buttons_info[button_name]["button"] = button

            # kind = button_info["kind"]
            func = button_info["function"]

            button.Bind(wx.EVT_BUTTON, func)

            # Add row to main vertical sizer
            outer_sizer.Add(row_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        # === Add a stretch spacer and exit button at the bottom ===
        outer_sizer.AddStretchSpacer()

        exit_button = wx.Button(panel, label="Exit")
        exit_button.Bind(wx.EVT_BUTTON, self.on_exit)
        outer_sizer.Add(exit_button, 0, wx.ALIGN_CENTER | wx.ALL, 10)

        panel.SetSizer(outer_sizer)
        self.Show()

    def on_exit(self, event):
        self.Close()

    def on_open_csv(self, kind):
        with wx.FileDialog(
            self,
            f"Open {kind} CSV file",
            wildcard="CSV files (*.csv)|*.csv",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as file_dialog:
            if file_dialog.ShowModal() == wx.ID_CANCEL:
                return
            path = file_dialog.GetPath()
            if kind == "train":
                self.buttons_info["TrainingData"]["label"].SetLabel(
                    f"Training CSV Loaded: {path}"
                )
            elif kind == "test":
                self.buttons_info["TestData"]["label"].SetLabel(
                    f"Test CSV Loaded: {path}"
                )

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
        # Placeholder for EDA functionality
        self.buttons_info["StartEDA"]["label"].SetLabel(
            "EDA started (placeholder)"
        )

    def make_placeholder_handler(self, index):

        self.buttons_info[f"Button{index}"]["label"].SetLabel(
            f"Button {index} clicked"
        )
