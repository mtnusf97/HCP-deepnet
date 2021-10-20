import torch
import torch.nn as nn

__all__ = ['LSTM', 'classification_loss', 'regression_loss']


class LSTMLayer(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=4, batch_first=True,
                            dropout=0.5, bidirectional=True)

    def forward(self, x):
        # x: batch_size * connections * window_size
        x = torch.transpose(x, 1, 2)
        x = self.lstm(x)
        # x[0]: batch * window_size * hidden_size
        return x


class LSTM(nn.Module):

    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config
        self.window_size = config.model.window_size
        self.rnn_input_size = config.model.rnn_input_size
        self.rnn_hidden_size = config.model.rnn_hidden_size
        self.lstm_layer = LSTMLayer(self.rnn_input_size, self.rnn_hidden_size)
        self.avg_pooling = nn.AvgPool1d(kernel_size=self.window_size)
        self.fc = nn.Linear(in_features=2*self.rnn_hidden_size, out_features=1)

    def forward(self, x):
        # x: batch_size * connections * window_size
        x = self.lstm_layer(x)[0]
        # x: batch_size * window_size * rnn_hidden_size
        x = torch.transpose(x, 1, 2)
        # x: batch_size * rnn_hidden_size * window_size
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
