import logging

LOG_FORMAT = "[%(levelname)s] <%(name)s>: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a consistent format for all CLI entry points."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
