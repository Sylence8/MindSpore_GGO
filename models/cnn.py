import mindspore.nn as nn
from mindspore.common.initializer import Normal

class C3D(nn.Cell):
    def __init__(self, num_classes=600, in_channels=1):
        super(C3D, self).__init__()

        self.group1 = nn.SequentialCell(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        self.group2 = nn.SequentialCell(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.group3 = nn.SequentialCell(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.group4 = nn.SequentialCell(
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.group5 = nn.SequentialCell(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        )
        
        last_duration = 16
        last_size = 112 // 32
        self.fc1 = nn.SequentialCell(
            nn.Dense(512 * last_duration * last_size * last_size, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc2 = nn.SequentialCell(
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Dense(4096, num_classes)
    
    def construct(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    import numpy as np
    import mindspore.context as context
    from mindspore import Tensor
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    
    model = C3D(num_classes=600, in_channels=1)
    print(model)
    
    input_var = Tensor(np.random.randn(8, 1, 16, 112, 112).astype(np.float32))
    output = model(input_var)
    print(output.shape)
