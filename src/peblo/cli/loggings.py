# coding=utf-8
import logging
import sys


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",    # Cyan
        logging.INFO: "\033[32m",     # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",    # Red
        logging.CRITICAL: "\033[41m", # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}"


def setup_logging(debug: bool = False, log_file: str | None = None):
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.handlers.clear()

    # terminal handler（with colors）
    console = logging.StreamHandler()
    if sys.stderr.isatty():
        console.setFormatter(ColorFormatter(
            "%(levelname)s:%(name)s:%(message)s"
        ))
    else:
        console.setFormatter(logging.Formatter(
            "%(levelname)s:%(name)s:%(message)s"
        ))
    root.addHandler(console)

    # file handler（without color）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        ))
        root.addHandler(file_handler)
