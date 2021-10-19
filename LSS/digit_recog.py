import torch
import torch.nn as nn
import torch.functional as F
from torchvision import datasets,transforms
from tqdm import tqdm
import numpy as np






class Digit_rec(nn.Module):
    def __init__(self,in_channels,img_size):
        super(Digit_rec,self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.Linear_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128*(img_size//4)*(img_size//4),128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x = self.conv_block(x)
        x = x.view(x.size(0),-1)
        x = self.Linear_block(x)
        return x


def prep_dataset(dir_root,transform,download=True,mode=0):
    if mode == 0:        # 0 for training dataset
        dataset = datasets.MNIST(dir_root,download=download,train=True,transform=transform)
    elif mode == 1:      # 1 for testing dataset
        dataset = datasets.MNIST(dir_root,download=download,train=False,transform=transform)
    return dataset

def prep_dataloader(batch_size,dataset):
    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

def save_model(model,file_name):
    try:
        torch.save(model.state_dict(),file_name)
        return True
    except [IOError,PermissionError]:
        print("permission error while writing a file")
        return False

def train(root_dir,batch_size,model,epochs = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])
    train_dataset = prep_dataset(root_dir+'/train',transform,True,0)
    train_loader = prep_dataloader(batch_size,train_dataset)
    loss = None
    
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(),1e-3)
    for epoch in range(epochs):
        model.train()
        loop =tqdm(train_loader,"Training : epoch {}".format(epoch+1))
        loop.set_postfix(loss=loss)
        for imgs,labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optim.zero_grad()
            output = model(imgs)
            loss_val = loss_fn(output,labels)
            loss = loss_val.item()
            loop.set_postfix(loss=loss)
            loss_val.backward()
            optim.step()
        test(model,root_dir,batch_size,epoch)
        print("loss of {} is {}".format(epoch+1,loss))
    save_model(model,'digit_model.h5')
def test(model,root_dir,batch_size,epoch_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])
    test_dataset = prep_dataset(root_dir+'/test',transform,True,0)
    test_loader = prep_dataloader(batch_size,test_dataset)  
    correct = 0
    model.eval()
    
    for imgs,labels in test_loader:
      imgs = imgs.to(device)
      labels = labels.to(device)
      output = model(imgs)
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
    print("Accuracy of epoch {} is  {}".format(epoch_num+1,100*(correct/len(test_loader.dataset))))





def test():
    d = Digit_rec(3,16,[4,8],3,2,1)
    data = torch.randn((2,3,28,28))
    print(d(data))

#test()
# model = Digit_rec(1,64,[16,32],3,2,1,28)
# train('dataset',64,model)