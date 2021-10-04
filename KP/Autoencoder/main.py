import torch
import torch.nn
import argparse
import math
import numpy as np
from tqdm import tqdm
from preprocess import MyDataset
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from model import Autoencoder
import matplotlib.pyplot as plt

hyperparams = {
     "batch_size": 100,
     "num_epochs": 1000,
     "learning_rate": 0.0001,
     "a1": 1,
     "a2": 2,
     "a3": 1,
     "a4": 1
 }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Train the Model
def train(model, train_loader):
    """
    Trains the model.
    """
    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # set model to train
    model = model.train()
    
    # record the loss
    loss_list = []
    step = 0
    
    # start training
    for sample in tqdm(train_loader):
        X = sample['perm_img'].float()
        Y = sample['cap_pair'].float()
                
        # zero optimizer
        optimizer.zero_grad()
        
        # 4 predictions and 4 loss
        enc_in_enc_out, dec_in_dec_out, enc_in_dec_out, dec_in_enc_out = model(X, Y)
        
        # print('---')
        # print(X.size())
        # print(dec_in_dec_out.size())
        
        loss_1 = loss_fn(enc_in_enc_out, Y)
        loss_2 = loss_fn(dec_in_dec_out, X)
        loss_3 = loss_fn(enc_in_dec_out, X)
        loss_4 = loss_fn(dec_in_enc_out, Y)
        
        loss = hyperparams["a1"] * loss_1 + hyperparams["a2"] * loss_2 + hyperparams["a3"] * loss_3 + hyperparams["a4"] * loss_4
        
        # backword and optimize
        loss.backward()
        optimizer.step()
        
        # print loss
        loss_cur = np.round(loss.detach().cpu().numpy(), 2)
        loss_list.append(loss_cur)
        step += 1
        # print('training step '+str(step)+" loss: " + str(loss_cur))

    return loss_list

# Test the Model
def test(model, test_loader):
    """
    Test the model.
    """
    # Define loss function
    loss_fn = nn.MSELoss()
    
    # set model to train
    model = model.eval()
    
    # record the loss
    loss_list = []
    step = 0
    
    # start testing
    for sample in tqdm(test_loader):
        X = sample['perm_img'].float()
        Y = sample['cap_pair'].float()
        
        # 4 predictions and 4 loss
        enc_in_enc_out, dec_in_dec_out, enc_in_dec_out, dec_in_enc_out = model(X, Y)
        
        loss_1 = loss_fn(enc_in_enc_out, Y)
        loss_2 = loss_fn(dec_in_dec_out, X)
        loss_3 = loss_fn(enc_in_dec_out, X)
        loss_4 = loss_fn(dec_in_enc_out, Y)
        
        loss = hyperparams["a1"] * loss_1 + hyperparams["a2"] * loss_2 + hyperparams["a3"] * loss_3 + hyperparams["a4"] * loss_4
        
        # print loss
        loss_cur = np.round(loss.detach().cpu().numpy(), 2)
        loss_list.append(loss_cur)
        step += 1
    
    loss_avg = np.mean(np.asarray(loss_list))
    
    return loss_avg


# Valid the Model
def valid(model, valid_loader, valid_dir):
    """
    predict the patterns
    """
    
    # start validating
    for sample in tqdm(valid_loader):
        X = sample['perm_img'].float()
        Y = sample['cap_pair'].float()
        name = sample['name']
        print(name)
        
        # prediction
        enc_in_enc_out, dec_in_dec_out, enc_in_dec_out, dec_in_enc_out = model(X, Y)
        X_pred = dec_in_dec_out.detach().cpu().numpy()
        
        # reshape
        X = X.reshape(50,84)
        X_pred = X_pred.reshape(50,84)
        
        # plot and save the figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(name[0])
        ax1.imshow(X, cmap='Greys_r')
        ax1.set_title('Original')
        
        ax2.imshow(X_pred, cmap='Greys_r')
        ax2.set_title('Predicted')        
        
        plt.savefig('compare'+name[0]+'.png')
        
        

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("test_dir")
    parser.add_argument("valid_dir")
    parser.add_argument("pretrained_model")
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store_true",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store_true",
                        help="run testing loop")
    parser.add_argument("-v", "--valid", action="store_true",
                        help="run validation loop")
    
    args = parser.parse_args()

    # fix the seed
    torch.manual_seed(0)
        
    # Load the dataset
    if args.train:
        train_dataset = MyDataset(args.train_dir, 'train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
        
    if args.test:
        test_dataset = MyDataset(args.test_dir, 'test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
       
    if args.valid:
        valid_dataset = MyDataset(args.valid_dir, 'valid')
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)    
    
    # iniaitlize the model
    model = Autoencoder()
    
    if args.load:
        print('\n loading pre-trained model \n')
        model.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))

    for epoch in range(hyperparams['num_epochs']):    
        
        if args.train:
            # run train loop
            print("\n running training loop epoch " + str(epoch)+"\n")
            loss_list = train(model, train_loader)
     
        if args.test:
            # run test loop
            print("\n running test loop epoch " + str(epoch)+"\n")
            loss_avg = test(model, test_loader)
            print("\n averaged test loss: " + str(loss_avg)+"\n")
            
        if args.save:
            # save every 5 epoch
            if (epoch+1) % 5 == 0:
                torch.save(model.state_dict(), 'model_3_bead/model_epoch_'+str(epoch)+'_loss_'+str(int(loss_avg*10000))+'.pt')
                
    if args.valid:
        # run validation loop
        valid(model, valid_loader, args.valid_dir)
