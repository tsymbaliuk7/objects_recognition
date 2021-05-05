from torch import nn
from torch import optim
import torch


class Model():
    def __init__(self, dataset, classes, max_epochs, learning_rate, momentum, loss_accuracy):
        self.model = nn.Sequential(nn.Linear(28 * 28, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.Sigmoid(),
                                   nn.Linear(128, 64),
                                   nn.Sigmoid(),
                                   nn.Linear(64, 5),
                                   nn.LogSoftmax(dim=1))
        self.dataset = dataset
        self.loss_method = nn.NLLLoss()
        self.max_epochs = max_epochs
        self.leaning_rate = learning_rate
        self.momentum = momentum
        self.loss_accuracy = loss_accuracy
        self.classes = classes


    def train(self):
        images = self.dataset.get_all_data()['images']
        labels = self.dataset.get_all_data()['labels']
        images = images.view(images.shape[0], -1)
        output = self.model(images)
        loss = self.loss_method(output, labels)
        loss.backward()
        optimizer = optim.SGD(self.model.parameters(), lr=self.leaning_rate, momentum=self.momentum)
        loss_prev = 0
        for e in range(self.max_epochs):
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = self.model(images)
            loss = self.loss_method(output, labels)
            loss.backward()
            optimizer.step()
            if abs(loss - loss_prev) <= self.loss_accuracy:
                print("Epoch N{0} with loss = {1};".format(e, loss))
                break
            loss_prev = loss
            if e % 500 == 0:
                print("Epoch N{0} with loss = {1};".format(e, loss))
        print('Training is ended!')

    def recognize(self, test_dataset):
        images = test_dataset.get_all_data()['images']
        labels = test_dataset.get_all_data()['labels']
        for i in range(len(labels)):
            image = images[i].view(1, 784)
            with torch.no_grad():
                output = self.model(image)
            ps = torch.exp(output)
            probability = list(ps.numpy()[0])
            predicted_label = probability.index(max(probability))
            true_label = labels.numpy()[i]
            print("Recognized: {:30} Actual: {:30}".format(list(self.classes[predicted_label].keys())[0],
                                                    list(self.classes[true_label].keys())[0]))
            test_dataset.show_image(i, 'Recognized: ' + list(self.classes[predicted_label].keys())[0])