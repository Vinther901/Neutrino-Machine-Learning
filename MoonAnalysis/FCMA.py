import numpy as np
from scipy.integrate import quad, quad_vec, romb
from iminuit.util import make_func_code
from iminuit import describe
from tqdm import tqdm

def sample(distribution,N):
    subsample = np.quantile(distribution,np.random.uniform(0,1,N))
    return subsample

def sample2d(df2d,N,noise=0.01):
    return df2d.sample(N,replace=True) + np.random.normal(0,noise,(N,2))

# def MCsample(func,xlims,ylims):
    

def fan_out(x):
    N = x.shape[0]
    angles = np.random.random(N)*2*np.pi
    tmp = np.zeros((N,2,2))
    sin = np.sin(angles)
    cos = np.cos(angles)
    tmp[:,0,0] = cos
    tmp[:,0,1] = -sin
    tmp[:,1,0] = sin
    tmp[:,1,1] = cos
    xy = np.hstack([x.reshape(N,1),np.zeros((N,1))])
    return np.matmul(xy.reshape(N,1,2),tmp).reshape(N,2)

def gauss(x,mu,sigma):
    x = np.asarray(x).reshape(-1,1)
    mu = np.expand_dims(mu,0)
    sigma = np.expand_dims(sigma,0)
    return np.exp(-0.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

class integratedLH:
    def __init__(self, f):
        self.f = f
        
        self.func_code = make_func_code(describe(self.f))
    
    def __call__(self, *par):
        return self.f(*par)
    
    def default_errordef(self):
        return 0.5

def log_gauss(x,mu,sigma,eps=1e-7):
    x = np.asarray(x).reshape(-1,1)
    mu = np.expand_dims(mu,0)
    sigma = np.expand_dims(sigma,0)
    return -0.5*((x-mu)/sigma)**2 - np.log(sigma*np.sqrt(2*np.pi)+eps)

def return_losses_1d(mus,sigmas,lims):
    def LF1_1d(mu, sigma):
        return -quad(lambda x: log_gauss(x,mu,sigma)*(gauss(x,mus,sigmas).sum(1)),*lims)[0]

    def LF2_1d(mu,sigma):
        return -np.sum(np.log(quad_vec(lambda x: gauss(x,mu,sigma)*gauss(x,mus,sigmas),*lims)[0]))
    return LF1_1d, LF2_1d

def gauss2d(x,y,mus,sigmas):
    xy = (np.vstack((x,y,)).T).reshape(-1,2,1)
    mus = mus.T.reshape(1,2,-1)
    sigmas = sigmas.reshape(1,-1)

    z_sqrd = (((xy - mus)/sigmas)**2).sum(1)
    return np.exp(-0.5*z_sqrd)/(2*np.pi*sigmas**2)

# def log_gauss2d(x,y,mu_x,mu_y,sigma,eps=1e-7):
#     z_sqrd = ((x-mu_x)**2 + (y - mu_y)**2)/ sigma**2
#     return -0.5*z_sqrd - np.log(2*np.pi*sigma**2 + eps)

# def return_LF1_2d_and_P_2d(xy_reco,xy_sigma,lim):
#     def P_2d(x,y,mu_x,mu_y,sigma,f):
#     #     z_sqrd = ((X.flatten()-mu_x)**2 + (Y.flatten() - mu_y)**2)/ sigma**2
#     #     P = 1/(4*int_lim**2 - f*sigma**2*2*np.pi)*(1 - f*np.exp(-0.5*z_sqrd))
#         z_sqrd = ((x-mu_x)**2 + (y - mu_y)**2)/ sigma**2
#         return 1/(4*lim**2 - f*sigma**2*2*np.pi)*(1 - f*np.exp(-0.5*z_sqrd))#np.log(1 - f*np.exp(-0.5*z_sqrd)/(2*np.pi*sigma**2))

#     def LF1_2d(mu_x,mu_y,sigma,f):
#         return -dblquad(lambda x,y: np.log(P_2d(x,y,mu_x,mu_y,sigma,f))*(gauss2d(x,y,xy_reco,xy_sigma).sum(1)),-lim,lim,-lim,lim)[0]
#     return LF1_2d, P_2d

def return_LF1_2d_and_P_2d(xy_reco, xy_sigma, N_samples, lim, subsample = 5000):

    assert np.log(N_samples - 1)/np.log(2)%1 == 0, "N_samples need to be of the form 2**k + 1"
    
    x_grid = np.linspace(-lim,lim,N_samples + 1)
    y_grid = x_grid.copy()
    
    X_grid, Y_grid = np.meshgrid(x_grid,y_grid)
    
    x = binc(x_grid)
    y = x.copy()
    
    X, Y = np.meshgrid(x,y)
    
    dx = (x[1:] - x[:-1]).mean()
    dy = (y[1:] - y[:-1]).mean()
    
    P_obs = np.zeros(N_samples**2)
    for i in tqdm(range(0,xy_reco.shape[0],subsample)):
        P_obs += gauss2d(X.flatten(),Y.flatten(),xy_reco[i:i+subsample],xy_sigma[i:i+subsample]).sum(1)
    
    def P_2d(x,y,mu_x,mu_y,sigma,f):
        z_sqrd = ((x-mu_x)**2 + (y - mu_y)**2)/ sigma**2
        return 1/(4*lim**2 - f*sigma**2*2*np.pi)*(1 - f*np.exp(-0.5*z_sqrd))
    
    def LF1_2d(mu_x,mu_y,sigma,f):
        P = np.log(P_2d(X.flatten(),Y.flatten(),mu_x,mu_y,sigma,f))*P_obs
        return -romb(romb(P.reshape(N_samples,N_samples,-1),dy,0),dx,0)
    return LF1_2d, P_2d, P_obs, (X_grid,Y_grid), (X,Y)

def return_LF2_2d_and_P_2d(xy_reco, xy_sigma, lim):
    def P_2d(x,y,mu_x,mu_y,sigma,f):
        Print("Wrong normalization!")
        z_sqrd = ((x-mu_x)**2 + (y - mu_y)**2)/ sigma**2
        return 1/(4*lim**2 - f*sigma**2*2*np.pi)*(1 - f*np.exp(-0.5*z_sqrd))
    
    def LF2_2d(mu_x,mu_y,sigma,f):
        Print("Called")
        loss = -np.sum(np.log(quad_vec(lambda y: quad_vec(lambda x: P_2d(x,y,mu_x,mu_y,sigma,f)*(gauss2d(x,y,xy_reco,xy_sigma)),-lim,lim)[0],-lim,lim)[0]))
        Print("Returning Loss")
        return loss
    return LF2_2d, P_2d

def binc(x):
    return 0.5*(x[1:] + x[:-1])

def Print(statement):
    from time import localtime, strftime
    print("{} - {}".format(strftime("%H:%M:%S", localtime()),statement))
                        
                     



























