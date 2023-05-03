#utils_semi.py
# Utils for semi-supervised learning
from utils.Hilbert_curve import HilbertCurve
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

'''
This code (idea) is from Hugo Gangloff's github repo:
'''

'''
1D sequence <-> image, using an hilbert curve
requires the sequence have length equal to a power of 2
based on hilbertcurve.py from the package https://github.com/galtay/hilbertcurve
'''

np.random.seed(19)

def create_missing_labels(img, p):
    '''
    p = probability of missing a pixel
    '''
    mask_missing = np.random.choice([0,1], size=img.shape, p=[p, 1-p])
    label_miss = img.copy()
    label_miss[mask_missing==0] = -1
    return label_miss

# def train(train_loader, epoch, model, optimizer, batch_sz, clip, print_every,device):
#     train_loss = 0
#     for batch_idx, datos in enumerate(train_loader):
#         x, y = datos
#         x, y = x.to(device), y.to(device)
#         x = x.transpose(0, 1).unsqueeze(2)
            
#         #forward + backward + optimize
#         optimizer.zero_grad()
    
#         kld_loss_l, rec_loss_l, y_loss_l, kld_loss_u, rec_loss_u, y_loss_u = model(data)
#         loss_l = kld_loss_l + rec_loss_l + y_loss_l
#         loss_u = kld_loss_u + rec_loss_u + y_loss_u
#         loss = loss_l + loss_u
#         loss.backward()
#         nn.utils.clip_grad_norm(model.parameters(), clip)
#         optimizer.step()
        
#         #printing
#         if batch_idx % print_every == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\t  Loss Labeled: {:.6f} \t Loss Unlabeled: {:.6f}'.format(
#                 epoch, batch_idx * len(datos), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader),
#                 loss_l.item()/batch_sz,
#                 loss_u.item()/batch_sz))     
#             print('\n kdl_loss_l: {:.4f} \t rec_loss_l: {:.4f} \t y_loss_l: {:.4f}'.format(kld_loss_l.item()/batch_sz, rec_loss_l.item()/batch_sz, y_loss_l.item()/batch_sz))
#             print('\n kdl_loss_u: {:.4f} \t rec_loss_u: {:.4f} \t y_loss_u: {:.4f}'.format(kld_loss_u.item()/batch_sz, rec_loss_u.item()/batch_sz, y_loss_u.item()/batch_sz))
#         train_loss += loss.item()
        
#     print('')
#     print('Train> Epoch: {} average -ELBO: {:.4f}'.format(epoch, train_loss/ len(train_loader.dataset)))
#     return train_loss/ len(train_loader.dataset)



def creation_noisy_image(img, mu, sigma,p,  size = 28):
    '''
    image: image of size 28*28
    Binary mask: 0 if the pixel is the background, 1 otherwise (part of the number)
    
    chain: chain of the image (Hilbert curve) which is a vector of size size*size 
    z =  N(0, 1) # random variable wich
    x = z* N(mu, sigma)

    return:
    image_x: image with noise (28*28)
    label_miss: input image with missing labels (28*28)
    '''
    # We use np.pad to add a border of 0 to the image and increase the size to 32*32 beacause:
    # 1D sequence <-> image, using an hilbert curve requires the sequence have length equal to a --power of 2--    
    # This is not a general solution, but it works for the MNIST dataset
    image = np.pad(img.reshape(size, size), ((2,2), (2,2)), 'constant')
    chain = image_to_chain(image)
    # More complex noisy image
    # mu*(chain[i]) is the mean of the normal distribution and it changes for each pixel with respect to the label at that pixel
    # z = np.random.normal(size = len(chain))
    # x = np.array([  z[i]*np.random.normal(mu*(chain[i]), sigma, 1) for i in range(len(chain)) ])
    x = np.array([np.random.normal(mu*(chain[i]), sigma, 1) for i in range(len(chain)) ])
    # In order to recuperate the original image we need to remove the first 2 and last 2 elements of the chain
    image_x = chain_to_image(x)[2:(2+size), 2:(2+size)]
    mask_missing = np.random.choice([0,1], size=(size, size), p=[p, 1-p])
    label_miss = img.reshape(size, size).copy()
    label_miss[mask_missing==0] = -1

    return [image_x, label_miss]

def semi_sup_preprocessing(list_images,p, mu, sigma, size):
    '''
    list_images: list of images (1, size*size)
    p: probability of a pixel to be missing
    mu: mean of the normal distribution
    sigma: std of the normal distribution

    return:
    x: list of noisy images of length size*size
    y: list of images with missing labels  of length size*size
    '''
    x_y = [creation_noisy_image(im, mu, sigma,p, size ) for im in list_images]
    return x_y
    #return [ x[0] for x in x_y], [ y[1] for y in x_y]



def dim_image(list_image, size = 28):
    '''
    This function is used to reshape the images specially for the particular case of the MNIST dataset
    list_image: list of images (1, size*size)
    return: list of images (size, size)
    '''
    return [img.reshape(size, size) for img in list_image]



def get_hilbertcurve_path(image_border_length):
    '''
    Given image_border_length, the length of a border of a square image, we
    compute path of the hilbert peano curve going through this image

    Note that the border length must be a power of 2.

    Returns a list of the coordinates of the pixel that must be visited (in
    order !)
    '''
    path = []
    p = int(np.log2(image_border_length))
    hilbert_curve = HilbertCurve(p, 2)
    path = []
    #print("Compute path for shape ({0},{1})".format(image_border_length,
        # image_border_length))
    for i in range(image_border_length ** 2):
        coords = hilbert_curve.coordinates_from_distance(i)
        path.append([coords[0], coords[1]])

    return path

def chain_to_image(X_ch, masked_peano_img=None):
    '''
    X_ch is an unidimensional array (a chain !) whose length is 2^(2*N) with N non negative
    integer.
    We transform X_ch to a 2^N * 2^N image following the hilbert peano curve
    '''
    if masked_peano_img is None:
        image_border_length = int(np.sqrt(X_ch.shape[0]))
        path = get_hilbertcurve_path(image_border_length)
        masked_peano_img = np.zeros((image_border_length, image_border_length))
    else:
        image_border_length = masked_peano_img.shape[0]
        path = get_hilbertcurve_path(image_border_length)
        
    X_img = np.empty((image_border_length, image_border_length))
    offset = 0
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_img[coords[0], coords[1]] = X_ch[idx - offset]
        else:
            offset += 1
            X_img[coords[0], coords[1]] = -1

    return X_img

def image_to_chain(X_img, masked_peano_img=None):
    '''
    X_img is a 2^N * 2^N image with N non negative integer.
    We transform X_img to a 2^(2*N) unidimensional vector (a chain !)
    following the hilbert peano curve
    '''
    path = get_hilbertcurve_path(X_img.shape[0])

    if masked_peano_img is None:
        masked_peano_img = np.zeros((X_img.shape[0], X_img.shape[1]))

    X_ch = []
    for idx, coords in enumerate(path):
        if masked_peano_img[coords[0], coords[1]] == 0:
            X_ch.append(X_img[coords[0], coords[1]])

    return np.array(X_ch)

