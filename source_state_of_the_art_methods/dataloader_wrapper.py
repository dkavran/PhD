from torch.utils.data import DataLoader

class DataloaderWrapper:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)