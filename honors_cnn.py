import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

torch.manual_seed(42)

transform = transforms.Compose(
    [transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

dataset = datasets.ImageFolder(root = './ajwaormedjool', transform = transform)

test_set, train_set = torch.utils.data.random_split(dataset, [0.2,0.8])
trainloader = torch.utils.data.DataLoader(train_set,batch_size=4,shuffle=False)
testloader = torch.utils.data.DataLoader(test_set,batch_size=4,shuffle=False)

learning_rate = 0.05
batch_size = 4
epoch_size = 40

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 5)
        self.conv3 = nn.Conv2d(10,20,5)
        self.fc1 = nn.Linear(1620, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

cnn.train() 
for epoch in range(epoch_size):

    loss = 0.0 

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = cnn(inputs)
        out = criterion(outputs, labels)
        out.backward()
        optimizer.step()
        
        loss +=  out.item()
        if i % 100 == 99:   
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 100:.3f}')
            loss = 0.0

print('Finished Training')

ground_truth = []
prediction = []
cnn.eval() 
with torch.no_grad(): 
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth += labels
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction += predicted
accuracy = accuracy_score(ground_truth,prediction)
recall = recall_score(ground_truth,prediction,average='weighted')
precision = precision_score(ground_truth,prediction,average='weighted')
print(accuracy)
print(recall)
print(precision)