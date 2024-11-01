from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.model import ResnetClassifier
        
class Tester:
    def __init__(self, model: ResnetClassifier, device, test_dataloader):
        self.model = model
        self.device = device
        self.test_dataloader = test_dataloader

    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(self.test_dataloader, desc='Testing'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax)
        plt.show()
        class_report = classification_report(all_labels, all_preds, digits=5)
        print("\nClassification Report:")
        print(class_report)