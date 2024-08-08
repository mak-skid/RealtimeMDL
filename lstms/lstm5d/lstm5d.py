import torch
import torch.nn as nn

class LSTM5D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bias=True):
        super(LSTM5D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.device = None

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim

            cell_list.append(_LSTM5DCell(input_dim=cur_input_dim,
                                      hidden_dim=self.hidden_dim,
                                      bias=bias))
        
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        '''
        Parameters
        ..........
        input_tensor (Tensor) - History sequence part of the input sample
        '''
        if self.device is None:
            self.device = input_tensor.device
        
        b, seq_len, channels, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, c, w)
        
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):     
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
        
        return layer_output, (h, c)

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width, self.device))
        return init_states
    
    
class _LSTM5DCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias):
        super(_LSTM5DCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.i = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.f = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.o = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.g = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        
    def forward(self, input_tensor, cur_state):
        '''
        Parameters
        ..........
        input_tensor (Tensor) - History sequence part of the input sample
        cur_state (Tuple) - A tuple of pair denoting the hidden state: (h, c)
        '''
        h_cur, c_cur = cur_state
        c = torch.cat([input_tensor, h_cur], dim=1)

        i = torch.sigmoid(self.i(c))
        f = torch.sigmoid(self.f(c))
        o = torch.sigmoid(self.o(c))
        g = torch.tanh(self.g(c))
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
    
    #(batch: 16, history_len:20, features:1, lane:5, section:21)
    # (lane, section, batch, history_len, feature)