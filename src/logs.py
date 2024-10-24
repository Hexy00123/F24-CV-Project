import logging
import os

def setup_logging(log_dir='Logs', log_filename='training.txt'):
    """
    Sets up logging for the project.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging is set up.")
    return logger
