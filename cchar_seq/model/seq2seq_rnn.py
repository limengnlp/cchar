import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout, model_type):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        if model_type == 'lstm':               
            self.rnn = nn.LSTM(input_size=emb_dim, 
                    hidden_size=hid_dim, 
                    num_layers=n_layers, 
                    batch_first=True, 
                    dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size=emb_dim, 
                    hidden_size=hid_dim, 
                    num_layers=n_layers, 
                    batch_first=True, 
                    dropout=dropout)
        else:
            raise ValueError("Only lstm or gru models are accepted.")      
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
              
        #src (extracted_feats) = [batch size, src len, emb dim]
        packed_embedded = pack_padded_sequence(src, src_len.to('cpu'), batch_first=True, enforce_sorted=False)                
        packed_output, hidden_state  = self.rnn(packed_embedded)
        output_padded, output_lengths = pad_packed_sequence(packed_output, batch_first=True)       
        #outputs = [batch size, src len, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size,  hid dim]
        
        return output_padded, hidden_state


class RNNDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, model_type):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        if model_type == 'lstm':         
            self.rnn = nn.LSTM(input_size=emb_dim, 
                    hidden_size=hid_dim, 
                    num_layers=n_layers, 
                    batch_first=True, 
                    dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size=emb_dim, 
                    hidden_size=hid_dim, 
                    num_layers=n_layers, 
                    batch_first=True, 
                    dropout=dropout)
        else:
            raise ValueError("Only lstm or gru models are accepted.")                 
        self.fc_out = nn.Linear(hid_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden_state):
        
        #input = [batch size]              
        input = input.unsqueeze(1)
        #input = [batch size, 1]       
        embedded = self.embedding(input)        
        #embedded = [batch size, 1, emb dim]

        output, hidden_state = self.rnn(embedded, hidden_state)            
        #seq len and n directions is 1 in the decoder:
        #output = [batch size, 1, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(1)) # 0 when batch_first=False       
        #prediction = [batch size, output dim]
        
        return prediction, hidden_state


class RNNSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, src_len):
        
        #src = [batch size, src len, img_feat]
        #src_len = [len]
        #trg = [batch size, trg len]        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        dec_outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)       
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        output, hidden_state = self.encoder(src, src_len)

        #first input to the decoder is the <bos> tokens
        input = trg[:, 0]   # [batch, t_steps]

        for t in range(1, trg_len):            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden_state = self.decoder(input, hidden_state)
            # output: [batch_size, output_dim]
            #place predictions in a tensor holding predictions for each token
            dec_outputs[:, t, :] = output
            
            #get the highest predicted token from our predictions
            input = torch.argmax(output, dim=1)     # output.argmax(1)          
        
        return dec_outputs

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)