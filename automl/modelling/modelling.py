from library import Logger


class AutoML_Modeling:
    def __init__(
        self,
        report_file: str,
        file_train: str,
        file_test: str = "",
        title: str = "",
        target: str = "",
        description: str = "",
        nogui=True,
        update_script: str = "",
        logger: Logger | None = None,
    ) -> None:
        self.report_file = report_file
        self.file_train = file_train
        self.file_test = file_test
        self.title = title
        self.target = target
        self.description = description
        self.nogui = nogui
        self.update_script = update_script
        if logger is None:
            self.logger = Logger(
                level_console=Logger.INFO,
                level_file=Logger.DEBUG,
                filename="",
                wx_handler=None,
            )
        else:
            self.logger = logger
