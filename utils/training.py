import torch
import torch.nn as nn
import numpy as np
import os
alpha = 0.1
beta = 0.1

def img_save(im, val = 1.001):
    n, m = im.shape
    img = im.copy()
    img[n-1,m-1] = val
    return img

def model_reconstruction(model,epoch_model, path_save, device, print_loss =True):
    print('Actual  path for to initialize our models: ', path_save)
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_'+str(epoch_model)+'.pth') 
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_model}')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if print_loss:
        print('loss: ',{checkpoint['loss']}, 'and epoch: ', {checkpoint['epoch']})
    return model

def num_param(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_loss_epoch(model, path_save,data, epoch_model):
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_train_'+str(epoch_model)+'.npy') 
    train_LOSS = np.load(path, allow_pickle=True)
    import matplotlib.pyplot as plt
    plt.plot(train_LOSS)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss of ' + data+ ' with ' + str(epoch_model) + ' epochs')
    plt.savefig(os.path.join(path_save, data + '_loss_' + str(epoch_model) +'.png'))
    plt.show()
    plt.close()
    return train_LOSS

def run_model_seq(x, y,model,optimizer,clip, path_save_model, n_epochs,save_every=5, print_every=1, epoch_init=1):
    train_LOSS = []
    path_save = os.path.join(path_save_model, model.__class__.__name__.casefold() +'_state_')
    print('The model is saved in this path', os.path.join(path_save_model, model.__class__.__name__.casefold()))
    for epoch in range(epoch_init, n_epochs + 1):
        #training
        # if  model.__class__.__name__.casefold() == 'svrnn':
        #     kld_loss_l, rec_loss_l, y_loss_l, kld_loss_u, rec_loss_u, y_loss_u = model(x,y)
        #     loss_l = kld_loss_l + rec_loss_l + y_loss_l
        #     loss_u = kld_loss_u + rec_loss_u + y_loss_u        
        #     loss = loss_l + loss_u
        if  model.__class__.__name__.casefold() == 'vsl':
            kld_loss_u, rec_loss_u, y_loss_l = model(x,y)
            loss_l = y_loss_l
            loss_u = kld_loss_u + rec_loss_u        
            loss = loss_l + beta*loss_u
        else:
            # 'svrnn_2' "tmm" all versions
            kld_loss_l, rec_loss_l, y_loss_l, kld_loss_u, rec_loss_u, y_loss_u, add_term= model(x,y)
            loss_l = kld_loss_l + rec_loss_l + y_loss_l
            loss_u = kld_loss_u + rec_loss_u + y_loss_u        
            loss = loss_l + loss_u + alpha*add_term

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), clip)
        optimizer.step()
        train_LOSS.append(loss.item())

        if epoch % print_every == 0:
            print('Loss Labeled: {:.6f} \t Loss Unlabeled: {:.6f}'.format(
                    loss_l, loss_u))     
            # print('\n kdl_loss_l: {:.4f} \t rec_loss_l: {:.4f} \t y_loss_l: {:.4f}'.format(kld_loss_l, rec_loss_l, y_loss_l))
            # print('\n kdl_loss_u: {:.4f} \t rec_loss_u: {:.4f} \t y_loss_u: {:.4f}'.format(kld_loss_u.item(), rec_loss_u.item(), y_loss_u.item()))
            
        if epoch % save_every == 0:
            fn = path_save+str(epoch)+'.pth'
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, fn)
            #torch.save(model.state_dict(), fn)
            # print('Saved model to '+fn)
            np.save(path_save+'train_'+str(epoch)+'.npy', train_LOSS)     
    return train_LOSS

def final_model(model, optimizer, epoch_model, path_save,device, print_loss =True):
    print('Actual  path for to initialize our models: ', path_save)
    path = os.path.join(path_save, model.__class__.__name__.casefold()+'_state_'+str(epoch_model)+'.pth') 
    print(path)
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path)
    print(f'Initialization of the {model.__class__.__name__} model  at epoch {epoch_model}')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if print_loss:
        print(f'loss: {loss} and epoch: {epoch}')
    return model