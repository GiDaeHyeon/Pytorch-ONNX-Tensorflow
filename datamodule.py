import pytorch_lightning as plfrom torch.utils.data import DataLoaderfrom torchvision.datasets import MNISTimport torchvision.transforms as transformsclass MNistDataModule(pl.LightningDataModule):    def __init__(self):        super(MNistDataModule, self).__init__()        self.train_dataset = MNIST(            root='./data/MNIST',            train=True,            download=True,            transform=transforms.Compose([                transforms.ToTensor()            ])        )        self.val_dataset = MNIST(            root='./data/MNIST',            train=False,            download=True,            transform=transforms.Compose([                transforms.ToTensor()            ])        )    def train_dataloader(self):        return DataLoader(            dataset=self.train_dataset,            batch_size=512,            drop_last=True,            pin_memory=True        )    def val_dataloader(self):        return DataLoader(            dataset=self.val_dataset,            batch_size=512,            drop_last=False,            pin_memory=True        )