import wx

class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title="Exit App")

        self.SetMinSize(wx.Size(400, 300))
        self.Maximize(True)

        panel = wx.Panel(self)

        # Create main vertical sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Add stretchable space to push the button to the bottom
        main_sizer.AddStretchSpacer()

        # Create horizontal sizer for the button row
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.AddStretchSpacer()
        exit_button = wx.Button(panel, label="Exit")
        button_sizer.Add(exit_button, 0, wx.ALIGN_CENTER)
        button_sizer.AddStretchSpacer()

        # Add the button sizer to the main sizer
        main_sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 20)

        panel.SetSizer(main_sizer)

        exit_button.Bind(wx.EVT_BUTTON, self.on_exit)

        self.Show()

    def on_exit(self, event):
        self.Close()

if __name__ == "__main__":
    app = wx.App(False)
    frame = MyFrame()
    app.MainLoop()
