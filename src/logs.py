import logging
import os
from datetime import datetime

def setup_logging(log_dir='Logs'):
    """
    Sets up logging for the project.
    """
    # Generate a timestamp-based filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'training_{timestamp}.txt'
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
