import os , shutil
from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
#from torch.optim import Adam


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # save loss and accuracy
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomAffine,ColorJitter

from  models import SimpleCNN
from animal_dataset import Animal_Dataset


def get_args():
    parser = ArgumentParser(description="CNN Training")
    parser.add_argument("--root","-r", type=str, default="dataset/animals", help="Root of the dataset")
    parser.add_argument("--epochs","-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size","-b", type=int, default=8, help="Batch Size")
    parser.add_argument("--image_size", "-i",type=int, default=224, help="Batch Size")
    parser.add_argument("--logging","-l",type=str, default="tensorboard")
    parser.add_argument("--trained_models","-m",type=str, default="trained_models")
    parser.add_argument("--checkpoint","-c",type=str, default=None)
    args= parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args::
    cm (array, shape = [n,n]): a confusion matrix of interger classes
    class_names (array, shape= [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(20,20)) 
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Normalize the confusion matrtix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)



if __name__ == '__main__':

    args = get_args()
    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

# https://pytorch.org/vision/main/auto_examples/others/plot_transforms.html#sphx-glr-auto-examples-others-plot-transforms-py

    train_transform = Compose([  
        RandomAffine(
            degrees = (-5,5),
            translate = (0.15, 0.15),
            scale=(0.85, 1.15),
            shear = 10),
        
        ColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.25,
            hue = 0.2),
        Resize((args.image_size,args.image_size)),
        ToTensor(),
        # Normalize(mean=[0.485, 0.456, 0.406],     # Dùng thêm khi dùng transfer learning
        #           std=[0.229,0.224,0.225])
    ])
    test_transform = Compose([  
        Resize((args.image_size,args.image_size)),
        ToTensor()])
    

    dataset_train = Animal_Dataset(root=args.root, train=True,transform = train_transform)
    
    dataset_val = Animal_Dataset(root=args.root, train=False,transform = test_transform)

    train_dataloader = DataLoader(dataset = dataset_train,
                                batch_size=args.batch_size , 
                                num_workers = 4, 
                                shuffle=True,
                                drop_last = True,)
    
    val_dataloader = DataLoader(dataset = dataset_val, 
                                batch_size = args.batch_size, 
                                num_workers = 4, 
                                shuffle=False,
                                drop_last = False,)
    
    if os.path.isdir(args.logging):
        #os.rmdir(args.logging) # chi xoa thu muc rong thoi
        shutil.rmtree(args.logging) # xao thu muc khong rong

    if not os.path.isdir(args.trained_models): # check thu co folder chua
        os.mkdir(args.trained_models)
    
    writer = SummaryWriter(args.logging)
    
    model = SimpleCNN(num_classes=10).to(device)

    # for name, param in model.named_parameters():
    #     if "fc." in name or "layer4." in name:
    #         pass
    #     else:
    #         param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    """
    model.parameters(): toàn bộ model được update weight
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    """
    Giờ muốn từ đầu đến giữa đóng băng, và train đoạn cuối
    """

    if args.checkpoint:  # not None
        checkpoint = torch.load(args.checkpoint)
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    else:
        start_epoch = 0
        best_acc = 0

    num_iter = len(train_dataloader)
    #best_acc = 0


    for epoch in range(start_epoch, args.epochs):

        model.train() # training model process
        progress_bar = tqdm(train_dataloader, colour="cyan")

        for iter,(images, labels) in enumerate (progress_bar):

         
            images = images.to(device); labels = labels.to(device)
            

            output = model(images) # forward process

            loss_value = criterion(output, labels) # tinh loss

            progress_bar.set_description("epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch + 1, args.epochs, iter+1, num_iter, loss_value))
           
            writer.add_scalar("Train/Loss",loss_value, epoch*num_iter + iter) # save loss

            optimizer.zero_grad() # clear buffer

            loss_value.backward() # computer gradients for backward

            optimizer.step() # update params


         # Model Evaluation process
        model.eval()   
        all_predictions = [] # Save predictions
        all_labels = [] # save labels

        for iter,(images, labels) in enumerate (val_dataloader): 
            all_labels.extend(labels)

    
            images = images.to(device); labels = labels.to(device)
                

            # quá trình test thì không update model, ko tính gradient
            with torch.no_grad(): 

                predictions = model(images) # prediction shape 16*10
                indices = torch.argmax(predictions.cpu(),dim=1) # dim=1 => 16 phan tử, dua ve cpu predictions.cpu()
                all_predictions.extend(indices)
                loss_value = criterion(predictions, labels)


# đưa từ tensor về scalar, lay cai nhan ben trong 
        all_labels = [label.item() for label in all_labels]
        all_predictions = [prediction.item() for prediction in all_predictions]
        
        plot_confusion_matrix(writer,confusion_matrix(all_labels,all_predictions), class_names=dataset_val.categories, epoch=epoch)

        accuracy = accuracy_score(all_labels, all_predictions)
        print("Epoch {} Accuracy: {}".format(epoch+1, accuracy))
        
        writer.add_scalar("Val/Accuracy",accuracy,epoch)

        #torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))

        checkpoint = {
            "epoch": epoch+1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() 
        }
        torch.save(checkpoint, "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() 
            }
            torch.save(checkpoint, "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy



"""
Nếu muốn gửi cho người khác thì gửi file best, còn muốn train tiếp thì dùng file last
        torch.save(model.state_dict(), "{}/last_cnn.pt".format(args.trained_models))
        if accuracy > best_acc:
            torch.save(model.state_dict(), "{}/best_cnn.pt".format(args.trained_models))
            best_acc = accuracy

Đây chỉ là cách save weights cơ bản, vì như vậy mình chưa save cả lr vd mai mình lấy train tiếp thì nó lấy cái lr to đùng
===> Tìm cách save cả mô hình và cả lr

python train_model.py --checkpoint trained_models/last_cnn.pt

"""