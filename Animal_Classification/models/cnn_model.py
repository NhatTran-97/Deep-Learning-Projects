import torch.nn as nn
import torch

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 1 block

        self.conv1 = self._make_block(in_channels=3, out_channels=8)
        self.conv2 = self._make_block(in_channels=8, out_channels=16)
        self.conv3 = self._make_block(in_channels=16, out_channels=32)
        self.conv4 = self._make_block(in_channels=32, out_channels=64)
        self.conv5 = self._make_block(in_channels=64, out_channels=128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=6272, out_features=512),
            nn.LeakyReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=num_classes),
            nn.LeakyReLU()
        )
        # for name, param in model.named_parameters():
        #     if "fc." in name or "layer4." in name:
        #         pass
        #     else:
        #         param.requires_grad = False

    
    def _make_block(self, in_channels, out_channels):
        return nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), # shape = 8*8*222*222
            nn.BatchNorm2d(num_features = out_channels),
            nn.LeakyReLU(), # shape van the

            # block 2
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1), # shape = 8*8*222*222
            nn.BatchNorm2d(num_features = out_channels),
            nn.LeakyReLU(), # shape van the
            nn.MaxPool2d(kernel_size=2)
        )


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        #x =self.flatten(x)
        #x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])

        x = x.view(x.shape[0],-1) # chi ra 1 chieu, con lai no tu gom

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    #model = SimpleNeuralNetwork()
    model = SimpleCNN()
    input_data = torch.rand(8,3,224,224)

    if torch.cuda.is_available():
        print(torch.cuda.is_available())
        model.cuda() # dua model vao trong cuda (in_place function)
        input_data = input_data.cuda()

    result = model(input_data)
    print(result.shape)