import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the project root directory from the environment variable
project_root = os.getenv('PROJECT_ROOT')
if not project_root:
    raise ValueError("PROJECT_ROOT environment variable is not set.")

# Define the runs directory
log_dir = os.path.join(project_root, 'training_logs/Logs')

# Ensure the runs directory exists
os.makedirs(log_dir, exist_ok=True)    

def setup_logging(log_dir=log_dir):
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
