import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

class BeamImageDataset(Dataset):
    """
    Custom dataset for beam training that includes target images and R-matrix parameters.
    """
    def __init__(self, target_data):
        """
        Initialize dataset with target data.
        
        Parameters:
        -----------
        target_data : list of dict
            Each dict should contain:
            {
                'image': numpy array or torch tensor,
                'r11': list or array,
                'r12': list or array, 
                'r33': list or array,
                'r34': list or array,
                'quad_k': list or array,
                'x_axis': optional,
                'y_axis': optional
            }
        """
        self.data = []
        
        for item in target_data:
            # Convert image to tensor if needed
            if isinstance(item['image'], np.ndarray):
                image = torch.tensor(item['image'], dtype=torch.float32)
            else:
                image = item['image'].clone().detach().float()
            
            # Convert R-matrix parameters to tensors
            r11 = torch.tensor(item['r11'], dtype=torch.float32)
            r12 = torch.tensor(item['r12'], dtype=torch.float32)
            r33 = torch.tensor(item['r33'], dtype=torch.float32)
            r34 = torch.tensor(item['r34'], dtype=torch.float32)
            quad_k = torch.tensor(item['quad_k'], dtype=torch.float32)
            
            self.data.append({
                'image': image,
                'r11': r11,
                'r12': r12,
                'r33': r33,
                'r34': r34,
                'quad_k': quad_k,
                'x_axis': item.get('x_axis', None),
                'y_axis': item.get('y_axis', None)
            })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def beam_collate_fn(batch):
    """
    Custom collate function to handle batching of beam data.
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack R-matrix parameters
    r11 = torch.stack([item['r11'] for item in batch])
    r12 = torch.stack([item['r12'] for item in batch])
    r33 = torch.stack([item['r33'] for item in batch])
    r34 = torch.stack([item['r34'] for item in batch])
    quad_k = torch.stack([item['quad_k'] for item in batch])
    
    return {
        'images': images,
        'r11': r11,
        'r12': r12,
        'r33': r33,
        'r34': r34,
        'quad_k': quad_k
    }

def create_beam_dataloader(target_data, batch_size=1, shuffle=True, num_workers=0):
    """
    Create a DataLoader for beam training.
    
    Parameters:
    -----------
    target_data : list of dict
        Target data with images and R-matrix parameters
    batch_size : int
        Batch size for training
    shuffle : bool
        Whether to shuffle the data
    num_workers : int
        Number of worker processes for data loading
        
    Returns:
    --------
    DataLoader
    """
    dataset = BeamImageDataset(target_data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        collate_fn=beam_collate_fn
    )

def split_dataset(dataset: BeamImageDataset, train_indices: list) -> tuple[Subset, Subset]:
    """
    Splits a BeamImageDataset into training and testing subsets based on provided indices.

    Parameters:
    -----------
    dataset : BeamImageDataset
        The full dataset to be split.
    train_indices : list
        A list of integer indices that should be used for the training set.

    Returns:
    --------
    tuple[torch.utils.data.Subset, torch.utils.data.Subset]
        A tuple containing (train_subset, test_subset).
    """
    # Convert train_indices to a set for efficient lookup
    train_indices_set = set(train_indices)

    # Determine all possible indices in the dataset
    all_indices = set(range(len(dataset)))

    # Calculate test_indices by finding the difference
    test_indices_set = all_indices - train_indices_set

    # Convert sets back to sorted lists to ensure determinism and for Subset constructor
    train_indices_list = sorted(list(train_indices_set))
    test_indices_list = sorted(list(test_indices_set))

    # Create Subset objects
    train_subset = Subset(dataset, train_indices_list)
    test_subset = Subset(dataset, test_indices_list)

    return train_subset, test_subset