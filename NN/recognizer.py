# import the required libraries
import torch
import os
import numpy
import io

import torchvision.io
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # neural network
import torch.nn.functional as F # raw function


# Neural network class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30000, 700)
        self.fc2 = nn.Linear(700, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return torch.sigmoid(x)

class Recognizer:
    category = {
        0: "nothing",
        1: "compost",
        2: "cardboard",
        3: "metal",
        4: "paper",
        5: "plastic_bottle"
    }

    def __init__(self, model_path):
            self.model = Net()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def imageProcess(self, image):
        preprocess = T.Compose([
            T.Resize(100),
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = preprocess(image)
        image = torch.flatten(image)
        return image

    def detect(self, image_path):
        img = Image.open(image_path)
        img = self.imageProcess(img)

        img = torch.flatten(img)
        return self.model(img)


    def train(self):
        dataset = []

        # load the images
        for i in range(6):
            item = self.category[i]
            for image in os.listdir("NN\\train\\" + item):
                if (image[-3:] == "png"):
                    try:
                        img = Image.open("NN\\train\\" + item + "\\" + image)

                        preprocess = T.Compose([
                            T.Resize(100),
                            T.CenterCrop(100),
                            T.ToTensor(),
                            T.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])

                        img = preprocess(img)
                        img = torch.flatten(img)

                        y = [0, 0, 0, 0, 0, 0]
                        y[i] = 1

                        y = torch.tensor(y, dtype=torch.float32)
                        data = [img, y]
                        dataset.append(data)
                    except Exception:
                        pass

        numpy.random.shuffle(dataset)

        # initialize the neural network
        net = Net()

        import torch.optim as optim

        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        EPOCH = 10
        BATCH_SIZE = 32
        # percent saved for validation
        VAL = 0.2

        test = dataset[:int(len(dataset) * VAL)]
        train = dataset[int(len(dataset) * VAL):]

        print("Starting training")
        for epoch in range(10):  # 3 full passes over the data
            numpy.random.shuffle(train)
            batch_set = train[:BATCH_SIZE]


            for data in batch_set:  # `data` is a batch of data
                X, y = data  # X is the batch of features, y is the batch of targets.
                net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
                output = net(X)
                loss = F.binary_cross_entropy(output, y)  # calc and grab the loss value

                loss.backward()  # apply this loss backwards thru the network's parameters
                optimizer.step()  # attempt to optimize weights to account for loss/gradients
            print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines!

        right = 0
        total = 0

        # tester
        with torch.no_grad():
            for data in test:
                x, y = data
                output = net(x)

                output = self.getMax(output)
                expect = self.getMax(y)

                print(f"{output}       {expect}")

                if output == expect:
                    right += 1

                total += 1

            print(f"Accuracy Rate: {right / total}")

        torch.save(net.state_dict(), 'model.pt')

    def getMax(self, data):

        curPos = 0

        curMax = 0
        posMax = 0

        for i in data:
            if i > curMax:
                curMax = i
                posMax = curPos

            curPos += 1

        return posMax

