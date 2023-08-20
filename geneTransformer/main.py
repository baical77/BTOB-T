import time

import torch.nn
from comet_ml import Experiment
import torch.optim as optim
from torchsummary import summary
from Project import Project
from data import get_dataloaders
# from models import GeneT
from models import MultiomicsT
from utils import device
from poutyne.framework import Model
from poutyne.framework.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from callbacks import CometCallback
from logger import logging
import torchmetrics

# num_classes = 2
# weights = [2/105, 103/105] #as class distribution
# class_weights = torch.FloatTensor(weights).cuda()

if __name__ == '__main__':
    project = Project()
    # our hyperparameters
    params = {
        'lr': 0.001,
        'batch_size': 1,
        'epochs': 100,
        'model': 'GeneTransformer'
    }
    logging.info(f'Using device={device} 🚀')
    # everything starts with the data
    train_dl, val_dl, test_dl = get_dataloaders(
        project.data_dir / 'multiomics' / 'train',
        project.data_dir / 'multiomics' / 'val',
        batch_size=params['batch_size'],
        pin_memory=False,
        num_workers=4,
    )

    # define our comet experiment
    experiment = Experiment(api_key="ui16hFsEbgmHoSC81E1pYGEhB",
                            project_name="dl-pytorch-template", workspace="baical77")
    experiment.set_name("Gene Transformer Experiment (multiomics)")
    experiment.log_parameters(params)
    # create our special resnet18
    # genet = GeneT().to(device)
    genet = MultiomicsT().to(device)
    # print the model summary to show useful information
    logging.info(summary(genet, (14123, 4)))
    # define custom optimizer and instantiate the trainer `Model`
    optimizer = optim.Adam(genet.parameters(), lr=params['lr'])
    # loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    loss_func = torch.nn.CrossEntropyLoss()
    model = Model(genet, optimizer, loss_func,
                  batch_metrics=["accuracy"],
                  ).to(device)
    # model = Model(genet, optimizer, loss_func,
    #               batch_metrics=["accuracy"],
    #               epoch_metrics=[torchmetrics.AUROC(num_classes=num_classes)]).to(device)

    # usually you want to reduce the lr on plateau and store the best model
    callbacks = [
        ReduceLROnPlateau(monitor="val_acc", patience=5, verbose=True),
        ModelCheckpoint(str(project.checkpoint_dir /
                            f"{time.time()}-model-total.pt"), save_best_only=True, verbose=True),
        EarlyStopping(monitor="val_acc", patience=40, mode='max'),
        CometCallback(experiment)
    ]
    model.fit_generator(
        train_dl,
        val_dl,
        epochs=params['epochs'],
        callbacks=callbacks,
    )
    # get the results on the test set
    loss, test_acc = model.evaluate_generator(test_dl)
    logging.info(f'test_acc=({test_acc})')
    experiment.log_metric('test_acc', test_acc)

