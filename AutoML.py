import wx
from AutoML_GUI import AutoMLFrame
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning GUI"
    )
    parser.add_argument("--training_data", help="filename for training data")
    parser.add_argument("--test_data", help="filename for test data")
    args = parser.parse_args()
    args.project_root = Path(__file__).resolve().parent
    app = wx.App(False)
    frame = AutoMLFrame(args)
    app.MainLoop()
