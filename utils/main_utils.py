import torch
import torch.nn as nn
from my_custom_ai.utils.misc_utils import Config
from my_custom_ai.utils.train_utils import FunctionContainer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def print_confusion_matrix(model, loader, device, digits):
    predictions = []
    true_labels = []
    for idx, batch in enumerate(loader):
        x, y = batch
        output = model(x.to(device))
        b_predictions = torch.argmax(output, dim=-1)

        predictions.append(b_predictions)
        true_labels.append(y)

    predictions = torch.cat(predictions)
    true_labels = torch.cat(true_labels)

    cm = confusion_matrix(true_labels.tolist(), predictions.tolist())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=digits, yticklabels=digits)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.show()

class CentroidConfigs(Config):
    def __init__(self, digits, qb_rep, qb_pe, iterations, **kwargs):
        super(CentroidConfigs, self).__init__(**kwargs)
        self.digits = digits
        self.qb_rep = qb_rep
        self.qb_pe = qb_pe
        self.iterations = iterations

class CentroidContainer(FunctionContainer):
    def __init__(self, configs, **kwargs):
        super(CentroidContainer, self).__init__()
        self.configs = configs
        self.ce = nn.CrossEntropyLoss()

    def batch_extractor(self, batch, *args, **kwargs):
        img, y = batch
        return img.to(self.configs.device), y.to(self.configs.device)


    def loss_function(self, model_output, y, *args, **kwargs):
        loss = self.ce(model_output, y)
        return loss

    def validation_performance(self, model, loader, *args, **kwargs):
        total = 0
        corrects = 0
        scores = {}

        for idx, batch in enumerate(loader):
            batch, y = self.batch_extractor(batch)
            output = model(batch)
            predictions = torch.argmax(output, dim=-1)
            correct = torch.sum(predictions == y)

            total += y.size(0)
            corrects += correct.item()

        score = corrects / total
        scores['score'] = score
        return scores


    def test_performance(self, model, loader, pbar, *args, **kwargs):
        pass
