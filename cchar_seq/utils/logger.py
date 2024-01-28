import logging

def get_logger(log_file, log_name):
    # define logger and name
    logger = logging.getLogger(log_name)
    # record DEBUG upper level info
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler(log_file, mode='a', encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger