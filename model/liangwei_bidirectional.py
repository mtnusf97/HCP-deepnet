import torch
import torch.nn as nn

__all__ = ['LiangweiBidrectional', 'classification_loss', 'regression_loss']


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, max_pool_size):
        super(ConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=4,
                               padding=2)
        self.conv8 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=8,
                               padding=4)
        self.conv16 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=16,
                                padding=8)
        self.max_pool_size = max_pool_size
        self.maxpool = nn.MaxPool1d(self.max_pool_size)

    def forward(self, x):
        # x: batch_size * connections * windows
        x4 = self.conv4(x)
        x8 = self.conv8(x)
        x16 = self.conv16(x)
        x = torch.cat((x4, x8, x16), 1)
        x = self.maxpool(x)
        # x: batch_size * out_channels * (windows/maxpool_size)
        return x


class LSTMLayer(nn.Module):

    def __init__(self, input_size, hidden_size, window_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True,
                            dropout=0.5, bidirectional=True)

    def forward(self, x):
        # x: batch_size * out_channels * (window_size)
        x = torch.transpose(x, 1, 2)
        x = self.lstm(x)
        # x[0]: batch * window_size * hidden_size
        return x


class LiangweiBidrectional(nn.Module):

    def __init__(self, config):
        super(LiangweiBidrectional, self).__init__()
        self.config = config
        self.cnn_in_channels = config.model.cnn_in_channels
        self.cnn_out_channels = config.model.cnn_out_channels
        self.rnn_input_size = config.model.cnn_out_channels * 3
        self.rnn_hidden_size = config.model.rnn_hidden_size
        self.window_size = config.model.window_size
        self.max_pool_size = config.model.max_pool_size
        self.conv_layer = ConvLayer(self.cnn_in_channels, self.cnn_out_channels, self.max_pool_size)
        self.lstm_layer = LSTMLayer(self.rnn_input_size, self.rnn_hidden_size, self.window_size / self.max_pool_size)
        self.avg_pooling = nn.AvgPool1d(kernel_size=int(self.window_size / self.max_pool_size))
        self.fc = nn.Linear(in_features=2*self.rnn_hidden_size, out_features=1)

    def forward(self, x):
        # x: batch_size * connections * window_size
        x = self.conv_layer(x)
        # x: batch_size * (3*cnn_out_channels) * (window_size / max_pool_size)
        x = self.lstm_layer(x)[0]
        # x: batch_size * (window_size / max_pool_size) * rnn_hidden_size
        x = torch.transpose(x, 1, 2)
        # x: batch_size * rnn_hidden_size * (window_size / max_pool_size)
        x = self.avg_pooling(x)
        # x: batch_size * rnn_hidden_size * 1
        x = torch.transpose(x, 1, 2)

        # x: batch_size * 1 * rnn_hidden_size
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
