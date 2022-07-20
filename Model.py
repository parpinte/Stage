import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
# Creating a CNN class 
class net(nn.Module):
    def __init__(self):
      super(net, self).__init__()
      self.conv1 = nn.Conv2d(3, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

class model():
    def __init__(self):
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 0.005


    def get_data(self):
        
        transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
        train_set = datasets.ImageFolder('training_set', transform=transform)
        test_set = datasets.ImageFolder('test_set', transform=transform)

        train = DataLoader(train_set, batch_size=32, shuffle=True)
        test = DataLoader(test_set, batch_size=32, shuffle=True)    

        self.train, self.test = train, test


    def load_classes(self, folder): 
        with open(folder + "classes.txt", "r") as f:
            lines = f.readlines()
        tup1 = tuple(line.split() for line in lines)
        classes = []

        for item  in tup1: 
            classes.append(item[0])

        self.classes =  tuple(classes)

    def train_imshow(self):
        self.load_classes('class/')
        dataiter = iter(self.train)
        images, labels = dataiter.next()
        fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
        for i in range(5):
            ax = axes[i]
            ax.imshow(images[i].permute(1, 2, 0)) 
            ax.title.set_text(' '.join('%5s' % self.classes[labels[i]]))
        plt.show()

    def CNN_generation(self):
        self.load_classes('class/')
        self.net_cnn = net()

    def hyper_param(self):
        self.criterion = nn.CrossEntropyLoss()

        # Set optimizer with optimizer
        self.optimizer = torch.optim.SGD(model.net_cnn.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay, momentum = self.momentum)  

        self.Total_epoch = 20

    def start_training(self):
        
        for epoch in range(self.Total_epoch):
            for i, (images, labels) in enumerate(self.train):
                outputs = model.net_cnn(images)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f'epoch = {epoch} | loss = {loss}' )


if __name__ == '__main__':
   model = model()
   model.get_data()
   model.CNN_generation()
   model.hyper_param()
   model.start_training()
