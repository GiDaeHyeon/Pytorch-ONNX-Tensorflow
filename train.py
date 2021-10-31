from pytorch_lightning import Trainerfrom pytorch_lightning.loggers import TensorBoardLoggerfrom pytorch_lightning.callbacks import ModelCheckpointlogger = TensorBoardLogger(    save_dir='./logs',    name='MNIST',    default_hp_metric=False)checkpoint_callback = ModelCheckpoint(    monitor='val_loss',    dirpath='./CKPT',    filename='MNIST',    mode='min')trainer = Trainer(max_epochs=10,                  logger=logger,                  callbacks=[checkpoint_callback])if __name__ == '__main__':    from trainmodule import TrainModule    from datamodule import MNistDataModule    model = TrainModule()    datamodule_ = MNistDataModule()    trainer.fit(model=model,                datamodule=datamodule_)