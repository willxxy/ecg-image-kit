import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import warnings


from CreasesWrinkles.creases import get_creased
from ImageAugmentation.augment import get_augment

def generate_ecg_image(signal_data, lead_names=None, output_path=None, 
                       resolution=200, pad_inches=0, 
                       hw_text=False, wrinkles=False, augment=False,
                       fully_random=False, seed=None, **kwargs):
    """
    Generate an ECG image from signal data with optional augmentations.
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        ECG signal data with shape (num_leads, num_samples)
    lead_names : list
        List of lead names corresponding to the rows in signal_data
    output_path : str
        Path to save the output image. If None, returns the image without saving.
    resolution : int
        Resolution of the output image in DPI
    pad_inches : int
        Padding around the ECG plot
    hw_text : bool
        Add handwritten text to the image
    wrinkles : bool
        Add wrinkles and creases to the image
    augment : bool
        Apply image augmentations (noise, rotation, etc.)
    fully_random : bool
        Randomly decide which augmentations to apply
    seed : int
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters for customization
        
    Returns:
    --------
    str or PIL.Image.Image
        Path to the saved image or the image object if output_path is None
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Default lead names if not provided
    if lead_names is None:
        lead_names = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Ensure signal_data has the right shape
    if signal_data.shape[0] != len(lead_names):
        raise ValueError(f"Signal data has {signal_data.shape[0]} leads but {len(lead_names)} lead names were provided")
    
    # Extract parameters from kwargs with defaults
    columns = kwargs.get('columns', 4)
    show_grid = kwargs.get('show_grid', True)
    grid_color = kwargs.get('grid_color', 'red')
    add_lead_names = kwargs.get('add_lead_names', True)
    
    # Random parameters if requested
    if fully_random:
        wrinkles = random.choice([True, False])
        augment = random.choice([True, False])
    
    # Create a temporary directory for intermediate files if needed
    temp_dir = kwargs.get('temp_dir', os.path.join(os.getcwd(), 'temp'))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate the base ECG plot
    fig, axes = plt.subplots(figsize=(12, 8), dpi=resolution)
    
    # Calculate time axis (assuming 500 Hz sampling rate by default)
    sampling_rate = kwargs.get('sampling_rate', 500)
    time = np.arange(signal_data.shape[1]) / sampling_rate
    
    # Plot the ECG signals
    lead_plots = []
    offset = 0
    for i, lead in enumerate(lead_names):
        # Add offset to separate leads visually
        offset_signal = signal_data[i] + offset
        line, = axes.plot(time, offset_signal, label=lead)
        lead_plots.append(line)
        offset -= 3  # Adjust this value to control spacing between leads
    
    # Add grid if requested
    if show_grid:
        axes.grid(True, which='both', color=grid_color, linestyle='-', alpha=0.3)
        axes.minorticks_on()
    
    # Add lead names if requested
    if add_lead_names:
        for i, lead in enumerate(lead_names):
            axes.text(0, -i*3, lead, fontsize=10)
    
    # Customize plot appearance
    axes.set_yticks([])  # Hide y-axis ticks
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    
    # Save the base ECG image
    base_image_path = os.path.join(temp_dir, 'base_ecg.png')
    plt.tight_layout(pad=pad_inches)
    plt.savefig(base_image_path, dpi=resolution)
    plt.close(fig)
    
    # Load the image for augmentations
    current_image = base_image_path
    
    # Apply augmentations if available        
    # Wrinkles and creases
    if wrinkles:
        crease_angle = kwargs.get('crease_angle', 90)
        if not kwargs.get('deterministic_angle', False):
            crease_angle = random.choice(range(0, crease_angle+1))
            
        num_creases_vertically = kwargs.get('num_creases_vertically', 10)
        if not kwargs.get('deterministic_vertical', False):
            num_creases_vertically = random.choice(range(1, num_creases_vertically+1))
            
        num_creases_horizontally = kwargs.get('num_creases_horizontally', 10)
        if not kwargs.get('deterministic_horizontal', False):
            num_creases_horizontally = random.choice(range(1, num_creases_horizontally+1))
            
        current_image = get_creased(
            current_image,
            output_directory=temp_dir,
            ifWrinkles=True,
            ifCreases=True,
            crease_angle=crease_angle,
            num_creases_vertically=num_creases_vertically,
            num_creases_horizontally=num_creases_horizontally,
            bbox=kwargs.get('lead_bbox', False)
        )
    
    # Image augmentation (noise, rotation, etc.)
    if augment:
        noise = kwargs.get('noise', 50)
        if not kwargs.get('deterministic_noise', False):
            noise = random.choice(range(1, noise+1))
            
        crop = 0
        if not kwargs.get('lead_bbox', False):
            do_crop = random.choice([True, False])
            if do_crop:
                crop = kwargs.get('crop', 0.01)
                
        temp = kwargs.get('temperature', 40000)
        if random.choice([True, False]):  # blue_temp
            temp = random.choice(range(2000, 4000))
        else:
            temp = random.choice(range(10000, 20000))
            
        rotate = kwargs.get('rotate', 0)
        
        current_image = get_augment(
            current_image,
            output_directory=temp_dir,
            rotate=rotate,
            noise=noise,
            crop=crop,
            temperature=temp,
            bbox=kwargs.get('lead_bbox', False)
            )
    
    # Save or return the final image
    if output_path:
        # If the output path is a directory, create a filename
        if os.path.isdir(output_path):
            output_path = os.path.join(output_path, f"ecg_{random.randint(1000, 9999)}.png")
        
        # Copy the final image to the output path
        img = Image.open(current_image)
        img.save(output_path)
        return output_path
    else:
        # Return the image object
        return Image.open(current_image)