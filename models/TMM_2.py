import torch
from torch import nn
import torch.distributions.normal as Norm
import torch.distributions.kl as KL
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import math
from torch.autograd import Variable

EPS = torch.finfo(torch.float).eps 
c = - 0.5 * math.log(2*math.pi)

class TMM_2(nn.Module):
    '''
    Second version of the TMM model without the additional term for the loss function
    '''
    def __init__(self, x_dim, z_dim, y_dim, num_neurons, device):
        super(TMM_2,self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.y_dim = y_dim
        self.num_neurons = num_neurons
        self.Soft_threshold = nn.Sigmoid()
        # Prior p(z_t | z_{t-1}) = N (μt, σt)
        self.prior_z = nn.Sequential( nn.Linear(self.z_dim, self.num_neurons),
                                  nn.ReLU())
        self.prior_z_mean = nn.Linear(self.num_neurons, self.z_dim)
        
        self.prior_z_std = nn.Sequential( nn.Linear (self.num_neurons, self.z_dim), 
                                    nn.Softplus())
        # Prior p(y_t | y_{t-1}) = Cat (θt) 
        # We will use the sigmoid function to ensure that the logits are positive and only two classes(if not we can use the softmax function)
        self.prior_y = nn.Sequential( nn.Linear(self.y_dim, self.num_neurons ),
                                nn.ReLU())
        self.prior_y_proba = nn.Sequential(nn.Linear(self.num_neurons, self.y_dim),
                                      nn.Sigmoid())
        # q(y_t | x_t, z_{t}) = Cat (θt)
        self.q_y = nn.Sequential( nn.Linear(self.x_dim + self.z_dim, self.num_neurons),
                                nn.ReLU())
        self.q_y_proba = nn.Sequential(nn.Linear(self.num_neurons, self.y_dim),
                                      nn.Sigmoid())
        # Encoder
        # q(z_t |z_{t-1}, x_t) = N (μt, σt)
        self.enc = nn.Sequential( nn.Linear(self.z_dim + self.x_dim , self.num_neurons),
                                nn.ReLU())
        self.enc_mean = nn.Linear(self.num_neurons, self.z_dim)
        self.enc_std = nn.Sequential( nn.Linear (self.num_neurons, self.z_dim), 
                                    nn.Softplus())
        # Decoder
        # p(x_t |z_t, y_t) = N (μt, σt)
        self.dec = nn.Sequential( nn.Linear(self.y_dim + self.z_dim , self.num_neurons),
                                nn.ReLU(),
                                nn.Linear(self.num_neurons, self.num_neurons),
                                nn.ReLU())
        self.dec_mean = nn.Linear(self.num_neurons, self.x_dim)
        self.dec_std = nn.Sequential( nn.Linear (self.num_neurons, self.x_dim), 
                                    nn.Softplus())
        # Recurrence
        #self.rnn = nn.RNNCell( self.x_dim + self.z_dim + self.y_dim, self.h_dim, bias, nonlinearity='tanh')#nn.GRU( h_dim + x_dim + z_dim + y_dim , h_dim, n_layers)

    def encoder(self, x, z):
        enc = self.enc(torch.cat([x, z], 0))
        enc_mean = self.enc_mean(enc)
        enc_std = self.enc_std(enc)
        return enc_mean, enc_std
    
    def decoder(self, z, y):
        dec = self.dec(torch.cat([z, y], 0))
        dec_mean = self.dec_mean(dec)
        dec_std = self.dec_std(dec)
        return dec_mean, dec_std
    
    def get_cost_labeled(self, x, y, zt):
        # zt : z_{t-1}
        prior_zt = self.prior_z(zt)
        prior_zt_mean = self.prior_z_mean(prior_zt)
        prior_zt_std = self.prior_z_std(prior_zt)
        # Encoder 
        enc_mean, enc_std = self.encoder(x, zt)
        z_t = self._reparameterized_sample(enc_mean, enc_std)
        # Decoder 
        dec_mean, dec_std = self.decoder(z_t, y)
        # Loss
        kld_loss_l = self._kld_gauss(enc_mean, enc_std, prior_zt_mean, prior_zt_std)
        rec_loss_l = self._rec_gauss(x, dec_mean, dec_std)
        return kld_loss_l, rec_loss_l, z_t

    def forward(self, x, y):
        zt = torch.zeros(self.z_dim).to(self.device)
        kld_loss_l, rec_loss_l, y_loss_l, add_term =  4*[0]
        kld_loss_u, rec_loss_u, y_loss_u = 3*[0]
        for t in range(x.size(0)):
            # Prior
            if t==0:
                p_yt = self.prior_y_proba(self.prior_y(0*y[t-1]))
            else:
                p_yt = self.prior_y_proba(self.prior_y(y[t-1]))

            if y[t] != -1:
                y_t =  y[t].clone()
                kld_loss, rec_loss, z_t = self.get_cost_labeled(x[t],y_t, zt)
                kld_loss_l += kld_loss 
                rec_loss_l += rec_loss
                y_loss_l += self._nll_ber(p_yt, y_t)
                # q_yt = self.q_y_proba(self.q_y(torch.cat([x[t],z_t], 0)))
                # add_term += self._add_term_labeled(y_t, q_yt, p_yt)
            else:
                q_yt = self.q_y_proba(self.q_y(torch.cat([x[t], zt], 0)))
                y_t = self._reparameterized_sample_Gumbell(q_yt)
                # loss
                kld_loss, rec_loss, z_t = self.get_cost_labeled(x[t], y_t, zt)
                kld_loss_u += kld_loss
                rec_loss_u += rec_loss
                y_loss_u += self._kld_cat(p_yt, q_yt)
            zt = z_t.clone()  
            
        return kld_loss_l, rec_loss_l, y_loss_l, kld_loss_u, rec_loss_u, y_loss_u, add_term
        
    def _add_term_labeled(self, y, q, p):
        return torch.sum(y * torch.log(p*q) + (1-y) * torch.log((1-p)*(1-q)))

    def _nll_ber(self, mean, x):
        nll_loss = F.binary_cross_entropy(mean, x, reduction='sum')
        return nll_loss
    
    # def _nll_gauss(self, mean, std, x):
    #     return torch.sum(torch.log(std + EPS) + torch.log(2*torch.pi)/2 + (x - mean).pow(2)/(2*std.pow(2)))

    def _rec_gauss(self, x, mean, std):
        rec_loss = torch.sum(c + torch.log(std) + (x - mean)**2 / (2 * std**2))
        return rec_loss
    
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        norm_dis2 = Norm.Normal(mean_2, std_2)
        norm_dis1 = Norm.Normal(mean_1, std_1)
        kl_loss = torch.sum(KL.kl_divergence(norm_dis1, norm_dis2))
        return    kl_loss
    
    def _kld_cat(self, q, p):
        kl_loss = torch.sum(q * torch.log(q/p)+ (1-q) * torch.log((1-q)/(1-p)))
        return kl_loss
    
    def reset_parameters(self, stdv = 0.1):
        for weight in self.parameters():
            #weight.normal_(0, stdv)
            weight.data.normal_(0, stdv)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)
    
    def _reparameterized_sample_Gumbell(self, mean):
        """using std to sample"""
        eps = torch.rand(mean.size()).to(self.device)
        eps = Variable(eps)
        value = (torch.log(eps) - torch.log(1-eps) + torch.log(mean) - torch.log(1-mean)).to(self.device)
        return self.Soft_threshold(value)
    
    def sample(self, x, y):
        '''
        Complete image
        '''
        y_complete = y.clone()
        zt = torch.zeros(self.z_dim).to(self.device)
        for t in range(x.size(0)):
            if y[t] != -1:
                enc_mean, enc_std = self.encoder(x[t], zt)
                z_t = self._reparameterized_sample(enc_mean, enc_std)
                y_t =  y[t].clone()
            else:
                q_yt = self.q_y_proba(self.q_y(torch.cat([x[t], zt], 0)))
                l_x_t = Bernoulli(q_yt)
                y_t = l_x_t.sample() 
                y_complete[t] = y_t.item() 
                
                enc_mean, enc_std = self.encoder(x[t], zt)
                z_t = self._reparameterized_sample(enc_mean, enc_std)
            zt = z_t.clone()  

        return y_complete

