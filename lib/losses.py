import torch
import torch.nn as nn


def load_criterion(cfg):

    def get_criterion(name, class_weights):
        if name == 'crossentropy':
            print(f"CrossEntropy with class_weights: {class_weights}")
            return nn.CrossEntropyLoss(weight=class_weights)
        elif name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError()

    class_weights_train = cfg['training'].get('class_weights_train', None)
    class_weights_test = cfg['training'].get('class_weights_test', None)

    if class_weights_train:
        print(f"Using class weights {class_weights_train} for training")
        class_weights_train = torch.tensor(class_weights_train).float().to(cfg['training']['device'])
    else:
        class_weights_train = None

    if class_weights_test:
        print(f"Using class weights {class_weights_test} for testing")
        class_weights_test = torch.tensor(class_weights_test).float().to(cfg['training']['device'])
    else:
        class_weights_test = None

    train_criterion = get_criterion(cfg['training']['train_criterion'], class_weights_train)
    test_criterion = get_criterion(cfg['training']['test_criterion'], class_weights_test)

    return train_criterion, test_criterion