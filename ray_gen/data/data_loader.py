from .dataset.range_data import RangeData, ValidRangeData
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DataLoadBuilder:
    def build_train_loader(self):
        myDataset = RangeData()
        train_loader = DataLoaderX(
            myDataset, shuffle=True, batch_size=16, num_workers=4, drop_last=True
        )
        return train_loader

    def build_valid_loader(self):
        myDataset = ValidRangeData()
        train_loader = DataLoaderX(
            myDataset, shuffle=True, batch_size=1, num_workers=2, drop_last=True
        )
        return train_loader
