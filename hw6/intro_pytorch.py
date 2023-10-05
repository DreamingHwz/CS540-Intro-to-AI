import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    custom_transform=transforms.Compose([
        transforms.ToTensor(),  # converts a PIL Image or numpy.ndarray to tensor
        transforms.Normalize((0.1307,), (0.3081,))  #normalizes the tensor with a mean and standard deviation
        ])
    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)

    if (training is True):
        return  torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        return  torch.utils.data.DataLoader(test_set, batch_size = 64)



def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),  #28*28, 128
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        )
    
    return model



def train_model(model, train_loader, criterion, T):
    for i in range(T):
        print("Train Epoch: %d " % i, end = "")
        
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        loss_sum = 0.0
        accuracy = 0
        loader_size = 0

        for input, label in train_loader:
            batch_size = label.size()[0]
            
            opt.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            opt.step()

            prediction = torch.max(output.data, 1)[1]
            for k in range(batch_size):
                accuracy += (prediction[k] == label[k])
            loss_sum += batch_size * loss.item()
            loader_size += batch_size

        print("Accuracy: %d/%d(%.2f%%) Loss: %.3f" % (accuracy, loader_size, 100*accuracy/loader_size, loss_sum/loader_size))
    
    return
        
    

def evaluate_model(model, test_loader, criterion, show_loss = True):
    with torch.no_grad():
        loss_sum = 0.0
        accuracy = 0
        loader_size = 0

        for input, label in test_loader:
            batch_size = label.size()[0]

            output = model(input)
            loss = criterion(output, label)
            prediction = torch.max(output.data, 1)[1]

            for k in range(batch_size):
                accuracy += (prediction[k] == label[k])
            loss_sum += batch_size * loss.item()
            loader_size += batch_size

    if (show_loss is True):
        print("Average loss: %.4f" % (loss_sum/loader_size))
    print("Accuracy: %.2f%%" % (100 * accuracy/loader_size))

    return



def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    
    data = test_images[index]
    logits = model(data)
    prob = F.softmax(logits, dim=1).detach().numpy()[0]
    
    class_prob = dict(zip(class_names, prob))
    class_prob = sorted(class_prob.items(), key = lambda x: -x[1])
    for i in range(3):
        print("%s: %.2f%%" % (class_prob[i][0], class_prob[i][1]*100))
    
    return


if __name__ == '__main__':
    T = 5
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    model.train()
    train_model(model, train_loader, criterion, T)
    model.eval()
    evaluate_model(model, test_loader, criterion, True)
    pred_test = next(iter(test_loader))[0]
    predict_label(model, pred_test, 3)

    pass