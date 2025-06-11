import torch
from torch.utils.data import Dataset, Subset, DataLoader

# Example dataset
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create a sample dataset and subset
dataset = MyDataset(torch.arange(10))
subset_indices = [0, 2, 4, 6, 8]
subset = Subset(dataset, range(0, 10, 2))
iterator = iter(subset)
for i in range(5):
    print(next(iterator))
# Access elements using __getitem__
element_at_index_1 = subset.__getitem__(1)  # Returns element at index 1 in the subset (which is 2 in the original dataset)
print(f"Element at index 1: {element_at_index_1}")

element_at_indices_1_and_3 = subset[1]  # Returns elements at indices 1 and 3 in the subset
print(f"Elements at indices 1 and 3: {element_at_indices_1_and_3}")

