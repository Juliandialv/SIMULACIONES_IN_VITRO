import inspect
from datetime import datetime
from enum import Enum

from PySide6.QtCore import QObject, Signal


class LogLevel(Enum):
    INFO    = "#dcdcdc"
    SUCCESS = "#5CB85C"
    WARNING = "#f0ad4e"
    ERROR   = "#d9534f"


class _Logger(QObject):
    """Singleton. Emite señales que MainWindow escucha."""
    message = Signal(str, str, str, str)  # timestamp, tag, text, color

    def log(self, text: str, level: LogLevel = LogLevel.INFO, _depth: int = 1):
        frame = inspect.stack()[_depth]
        origin_file = frame.filename.split("\\")[-1].replace(".py", "")
        origin_func = frame.function
        tag = f"{origin_file}.{origin_func}"
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.message.emit(timestamp, tag, text, level.value)


# Instancia global — se importa directamente
logger = _Logger()