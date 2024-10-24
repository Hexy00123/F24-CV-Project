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

def log_attention_results(results, writer, step):
    """
    Logs attention maps for a given set of results.
    """
    for encoder_idx, encoder_data in enumerate(results['encoders']):
        # Log aggregated attention maps
        writer.add_image(f'Attention/Aggregated_Max_Encoder_{encoder_idx}', 
                         encoder_data['aggregated']['max'].unsqueeze(0), step)
        writer.add_image(f'Attention/Aggregated_Mean_Encoder_{encoder_idx}', 
                         encoder_data['aggregated']['mean'].unsqueeze(0), step)
        writer.add_image(f'Attention/Aggregated_Min_Encoder_{encoder_idx}', 
                         encoder_data['aggregated']['min'].unsqueeze(0), step)
        
        # Log individual heads' normalized attention maps
        for head_idx, head_data in enumerate(encoder_data['heads']['normalized']):
            writer.add_image(f'Attention/Normalized_Head_{head_idx}_Encoder_{encoder_idx}', 
                             head_data.unsqueeze(0), step)
