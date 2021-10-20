import torch
import torch.nn as nn

__all__ = ['CNN', 'classification_loss', 'regression_loss']


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels_l1, out_channels_l2, out_channels_l3, max_pool_size):
        super(ConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels_l1 = out_channels_l1
        self.max_pool_size = max_pool_size
        self.out_channels_l2 = out_channels_l2
        self.out_channels_l3 = out_channels_l3

        self.conv_l1_4 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels_l1, kernel_size=4,
                                   padding=2)
        self.conv_l1_8 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels_l1, kernel_size=8,
                                   padding=4)
        self.conv_l1_16 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels_l1, kernel_size=16,
                                    padding=8)
        self.maxpool = nn.MaxPool1d(self.max_pool_size)
        self.conv_l2 = nn.Conv1d(in_channels=3*self.out_channels_l1, out_channels=self.out_channels_l2, kernel_size=8,
                                 padding=2)
        self.conv_l3 = nn.Conv1d(in_channels=self.out_channels_l2, out_channels=self.out_channels_l3, kernel_size=4,
                                 padding=2)

    def forward(self, x):
        # x: batch_size * connections * windows
        x4 = self.conv_l1_4(x)
        x8 = self.conv_l1_8(x)
        x16 = self.conv_l1_16(x)
        x = torch.cat((x4, x8, x16), 1)
        x = self.maxpool(x)
        # x: batch_size * out_channels * (windows/maxpool_size)
        x = self.conv_l2(x)
        x = self.conv_l3(x)
        return x


class CNN(nn.Module):

    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn_in_channels = config.model.cnn_in_channels
        self.cnn_out_channels_l1 = config.model.cnn_out_channels_l1
        self.cnn_out_channels_l2 = config.model.cnn_out_channels_l2
        self.cnn_out_channels_l3 = config.model.cnn_out_channels_l3
        self.window_size = config.model.window_size
        self.max_pool_size = config.model.max_pool_size
        self.conv_layer = ConvLayer(self.cnn_in_channels, self.cnn_out_channels_l1, self.cnn_out_channels_l2,
                                    self.cnn_out_channels_l3, self.max_pool_size)
        self.flatten = torch.flatten

        fc_in_features = ((int(self.window_size/self.max_pool_size) - 2) * self.cnn_out_channels_l3)
        self.fc = nn.Linear(in_features=fc_in_features, out_features=1)

    def forward(self, x):
        # x: batch_size * connections * window_size
        x = self.conv_layer(x)
        # x: batch_size * self.cnn_out_channels_l3 * (int(self.window_size/self.max_pool_size) - 2)
        x = self.flatten(x, 1, 2)
        # x: batch_size * (self.cnn_out_channels_l3 * (int(self.window_size/self.max_pool_size) - 2))
        x = self.fc(x)
        # x: batch_size * 1 * 1
        x = x.view(-1, 1)  # final score of the model
        # x: batch_size * 1
        return x


def classification_loss(model_output, labels):
    # model_output: batch_size * 1
    # labels: batch_size * 1
    criterion = nn.BCEWithLogitsLoss()
    return criterion(model_output, labels)


def regression_loss(model_output, target):
    # model_output: batch_size * 1
    # target: batch_size * 1
    criterion = nn.MSELoss()
    return criterion(model_output, target)
