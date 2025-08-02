import wx
from automl_gui import AutoMLFrame
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Machine Learning GUI"
    )
    parser.add_argument("--title", help="title of the project", default="")
    parser.add_argument("--training_data", help="filename for training data")
    parser.add_argument("--test_data", help="filename for test data")
    parser.add_argument(
        "--EDA",
        help="start EDA, default is False",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--nogui",
        help="execute without GUI, default is False",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--silent",
        help="if executed without GUI, no console output, default is False",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--report_file", help="filename for report file (.html)"
    )
    parser.add_argument("--output_file", help="filename for output file (.csv)")
    args = parser.parse_args()
    args.project_root = Path(__file__).resolve().parent
    app = wx.App(False)
    frame = AutoMLFrame(args)
    app.MainLoop()
