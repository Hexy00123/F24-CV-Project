import logging
import os
from datetime import datetime
from dotenv import load_dotenv
import matplotlib.pyplot as plt 
import torch 
from torchvision.utils import make_grid

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


def apply_cmap(attn): 
        cmap = plt.get_cmap('YlGnBu_r')
        colored_attn = cmap(attn).transpose(2, 0, 1)
        colored_attn = torch.tensor(colored_attn)
        return colored_attn[:3, :, :]


def log_attention_results(logs, writer, step):
    """
    Logs attention maps for a given set of results.
    """
    n_heads = logs['n_heads'] 
    results = logs['results']
    val_id = logs['val_id']
    orig = logs['orig']
    
    attentions = [] 
    for head in range(n_heads): 
        attn = results['encoders'][-1]['heads']['normalized'][head].detach().cpu()
        attentions.append(apply_cmap(attn))
        
    aggregations = [] 
    for k in results['encoders'][-1]['aggregated'].keys(): 
        attn = results['encoders'][-1]['aggregated'][k].detach().cpu()
        aggregations.append(apply_cmap(attn))
            
    grid = make_grid(torch.stack([orig] + aggregations + attentions), nrow=4)
    
    writer.add_image(f'Attention/val_img_{val_id+1}', grid, step)
