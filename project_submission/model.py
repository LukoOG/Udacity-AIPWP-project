import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict


def calculate_validation_loss(model, batch_num, validloader, device, criterion):
    model.eval()
    model.to(device)
    total_loss = 0
    accuracy = 0
    
    val_set = iter(validloader)
    for _ in range(batch_num):
        images, labels = next(val_set)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        top_k, top_class = torch.exp(outputs).topk(1, dim=1)
        matches = top_class == labels
        accuracy += torch.mean(matches.type(torch.FloatTensor))
        
        total_loss+=loss.item()
    average_accuracy = accuracy/batch_num
    average_loss = total_loss/batch_num
    return average_accuracy, average_loss
        

def train(model, trainloader, validloader, device, criterion, optimizer, epochs):
    model.to(device)
    eval_interval = 10
    for e in range(epochs):
        model.train()
        train_loss = 0
        print(f"\n---------------epoch {e+1}---------------\n\n")
        for step, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            
            images, labels = images.to(device), labels.to(device)
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
            print(f"step {step} training loss: {loss.item():.3f}")
            if step % eval_interval == 0:
                with torch.no_grad():
                    val_accuracy, val_loss = calculate_validation_loss(model, 8, validloader, device, criterion)
                print(f"step {step} validation loss: {val_loss:.3f}\n Validation accuracy: {val_accuracy} ")
        else:
            print(f"total loss for epoch {e+1} is {train_loss/len(trainloader)}\n\n")


class FeedForward(nn.Module):
    def __init__(self, hidden_units, output):
        super().__init__()
        self.fc1 = nn.Linear(2048, hidden_units)
        self.fc2 = nn.Linear(hidden_units, output)
        self.dropout = nn.Dropout(p=0.18)
    def forward(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = self.fc2(h1)
        return F.log_softmax(h2, dim=1)
    
def construct_model(arch, hidden_units):
    print(arch)
    if hasattr(models, arch): 
        model = getattr(models, arch)(pretrained=True) 
    else:
        raise ValueError(f"Model '{arch}' not found in torchvision.models")
    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = FeedForward(hidden_units, 102)
    return model

def load_model(path):
    checkpoint = torch.load(path)
    model = construct_model(checkpoint['arch'], checkpoint['hidden_units'])
    model.fc.load_state_dict(checkpoint['state_dict'])
    return model