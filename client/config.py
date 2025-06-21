import logging.config
import os

def configure_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": "%(asctime)s:%(name)s [%(levelname)s] %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s:%(name)s [%(levelname)s] %(filename)s:%(lineno)d:%(funcName)s - %(message)s"
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO", # Default level for console output
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG", # Default level for file output (more detailed)
                "formatter": "detailed",
                "filename": os.path.join(log_dir, "app.log"),
                "encoding": "utf8"
            },
            "error_file": {
                "class": "logging.FileHandler",
                "level": "ERROR", # Only log ERROR and CRITICAL to this file
                "formatter": "detailed",
                "filename": os.path.join(log_dir, "errors.log"),
                "encoding": "utf8"
            }
        },

        "loggers": {
            "voice_assistant": {
                "handlers": ["console", "file", "error_file"],
                "level": "INFO",
                "propagate": False # Prevent propagation to root logger
            }
        },

        "root": {
            "handlers": ["console"],
            "level": "WARNING"
        }
    }
    logging.config.dictConfig(LOGGING_CONFIG)