from torch import nn
import torch


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # encoder layers
        self.encoder_fc1 = nn.Linear(50*84,1000)
        self.encoder_fc2 = nn.Linear(1000,500)
        self.encoder_fc3 = nn.Linear(500,42)
        
        # decoder layers
        self.decoder_fc1 = nn.Linear(42,500)
        self.decoder_fc2 = nn.Linear(500,1000)
        self.decoder_fc3 = nn.Linear(1000,50*84)        

        # relu layer
        self.relu = nn.ReLU()
        
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        

    def forward(self, x, y):
        
        # encoder input to encoder output
        enc_in_enc_out = self.dropout(x)
        enc_in_enc_out = self.encoder_fc1(enc_in_enc_out)
        enc_in_enc_out = self.relu(enc_in_enc_out)
        enc_in_enc_out = self.dropout(enc_in_enc_out)
        enc_in_enc_out = self.encoder_fc2(enc_in_enc_out)
        enc_in_enc_out = self.relu(enc_in_enc_out)
        enc_in_enc_out = self.dropout(enc_in_enc_out)
        enc_in_enc_out = self.encoder_fc3(enc_in_enc_out)
        
        # deecoder input to decoder output
        dec_in_dec_out = self.dropout(y)
        dec_in_dec_out = self.decoder_fc1(dec_in_dec_out)
        dec_in_dec_out = self.relu(dec_in_dec_out)
        dec_in_dec_out = self.dropout(dec_in_dec_out)
        dec_in_dec_out = self.decoder_fc2(dec_in_dec_out)
        dec_in_dec_out = self.relu(dec_in_dec_out)
        dec_in_dec_out = self.dropout(dec_in_dec_out)
        dec_in_dec_out = self.decoder_fc3(dec_in_dec_out)
        
        # encoder input to decoder output
        enc_in_dec_out = self.dropout(x)
        enc_in_dec_out = self.encoder_fc1(enc_in_dec_out)
        enc_in_dec_out = self.relu(enc_in_dec_out)
        enc_in_dec_out = self.dropout(enc_in_dec_out)
        enc_in_dec_out = self.encoder_fc2(enc_in_dec_out)
        enc_in_dec_out = self.relu(enc_in_dec_out)
        enc_in_dec_out = self.dropout(enc_in_dec_out)
        enc_in_dec_out = self.encoder_fc3(enc_in_dec_out)

        enc_in_dec_out = self.dropout(enc_in_dec_out)
        enc_in_dec_out = self.decoder_fc1(enc_in_dec_out)
        enc_in_dec_out = self.relu(enc_in_dec_out)
        enc_in_dec_out = self.dropout(enc_in_dec_out)
        enc_in_dec_out = self.decoder_fc2(enc_in_dec_out)
        enc_in_dec_out = self.relu(enc_in_dec_out)
        enc_in_dec_out = self.dropout(enc_in_dec_out)
        enc_in_dec_out = self.decoder_fc3(enc_in_dec_out)  
        
        # decoder input to encoder output
        dec_in_enc_out = self.dropout(y)
        dec_in_enc_out = self.decoder_fc1(dec_in_enc_out)
        dec_in_enc_out = self.relu(dec_in_enc_out)
        dec_in_enc_out = self.dropout(dec_in_enc_out)
        dec_in_enc_out = self.decoder_fc2(dec_in_enc_out)
        dec_in_enc_out = self.relu(dec_in_enc_out)
        dec_in_enc_out = self.dropout(dec_in_enc_out)
        dec_in_enc_out = self.decoder_fc3(dec_in_enc_out)      
        
        dec_in_enc_out = self.dropout(dec_in_enc_out)
        dec_in_enc_out = self.encoder_fc1(dec_in_enc_out)
        dec_in_enc_out = self.relu(dec_in_enc_out)
        dec_in_enc_out = self.dropout(dec_in_enc_out)
        dec_in_enc_out = self.encoder_fc2(dec_in_enc_out)
        dec_in_enc_out = self.relu(dec_in_enc_out)
        dec_in_enc_out = self.dropout(dec_in_enc_out)
        dec_in_enc_out = self.encoder_fc3(dec_in_enc_out)        

        return torch.squeeze(enc_in_enc_out, 1), torch.squeeze(dec_in_dec_out, 1), torch.squeeze(enc_in_dec_out, 1), torch.squeeze(dec_in_enc_out, 1)