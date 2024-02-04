import torch
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import numpy as np
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("mps")

# net = torchvision.models.resnet34(weights='IMAGENET1K_V1')
# net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# net = torchvision.models.resnet18(weights='IMAGENET1K_V1')
# net = torchvision.models.resnet50(weights='IMAGENET1K_V1')

net = models.densenet121(weights='IMAGENET1K_V1')
net.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

print(net)
net.to(device)
print(next(net.parameters()).device)

def MyLoader(path):
    return Image.open(path).convert('L')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=MyLoader):
        with open(txt, 'r') as fh:
            imgs = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

root = r"png_state/"

train_data = MyDataset(txt=root + 'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=root + 'test.txt', transform=transforms.ToTensor())
test_num = len(test_data)

trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)
print('uploaded')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
train_accuracies = []
print("Start Training...")

patience = 10
n_folds = 10
num_epochs = 100

kfold = KFold(n_splits=n_folds, shuffle=True)
early_stop_counter = 0
best_loss = float('inf')

for fold, (train_ids, val_ids) in enumerate(kfold.split(trainloader.dataset)):
    print(f"FOLD {fold + 1}/{n_folds}")

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    trainloader_fold = torch.utils.data.DataLoader(trainloader.dataset, batch_size=16, sampler=train_subsampler)
    valloader_fold = torch.utils.data.DataLoader(trainloader.dataset, batch_size=16, sampler=val_subsampler)

    for epoch in range(num_epochs):
        print('-' * 30, '\n', 'epoch', epoch)
        net.train()
        loss100 = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader_fold):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            loss100 += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(valloader_fold):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valloader_fold)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print("Early stopping!")
            break

plt.plot(train_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()
print("done")

predicted_list = []
labels_list = []
dataiter = iter(testloader)

correct = 0
total = 0
net.eval()

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_list.extend(predicted.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())
        loss = criterion(outputs, labels)

confusion_mat = confusion_matrix(labels_list, predicted_list)
print('Accuracy : %f %%' % (100 * correct / total))
print('Loss : %f %%' % (loss.item()))
balanced_accuracy = balanced_accuracy_score(labels_list, predicted_list, adjusted=False)
print('Balanced Accuracy : %f %%' % (100 * balanced_accuracy))

f1_macro_score = f1_score(labels_list, predicted_list, average='macro')
print('F1 Macro : %f %%' % (100 * f1_macro_score))

con_mat_norm = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]  # 归一化
con_mat_norm = np.around(con_mat_norm, decimals=2)

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(con_mat_norm, annot=True, cmap='Blues')

ax.set(title='Confusion matrix',
       xlabel='Predicted label',
       ylabel='True label')
ax.set_xticklabels(['sleeping', 'other', 'bed2toilet', 'meal', 'realx', 'housekeeping', 'eating', 'wash_dish', 'leave',
                    'enter', 'work', 'respirate'])
ax.set_yticklabels(['sleeping', 'other', 'bed2toilet', 'meal', 'realx', 'housekeeping',
                    'eating', 'wash_dish', 'leave', 'enter', 'work', 'respirate'])

plt.show()
plt.savefig("confusion_matrix.pdf")
