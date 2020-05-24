import sys
import os
import logging
from logging import handlers

def configure_logging():
    LOG_FORMAT = '%(asctime)s :: %(name)-13s: %(levelname)-8s %(message)s'
    LOG_FILE = 'logs/run.log'
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    LOG_LEVEL = logging.INFO
    formater = logging.Formatter(LOG_FORMAT)
    logging.getLogger().setLevel(logging.NOTSET)

    # Add console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOG_LEVEL)
    console.setFormatter(formater)
    logging.getLogger().addHandler(console)

    # Add file rotating handler
    rotatingHandler = handlers.RotatingFileHandler(filename=LOG_FILE, maxBytes=1000000, backupCount=20)
    rotatingHandler.setLevel(LOG_LEVEL)
    rotatingHandler.setFormatter(formater)
    logging.getLogger().addHandler(rotatingHandler)

    suppress_import_modules = ["torch", "transformers"]
    for module_name in suppress_import_modules:
        logging.getLogger(module_name).setLevel(logging.WARNING)

    logging.info('Configured the logging successfully')

configure_logging()