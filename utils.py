import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 22})

def plot(epoch, model, loss, cuda, num_points, num_layers, num_hidden, x, y, lx, elx, lmbda):
    tlx, telx = torch.tensor(lx).float(), torch.tensor(elx).float()
    if cuda: tlx, telx = tlx.cuda(), telx.cuda()
    
    tlyhat, telyhat = model(tlx).view(-1), model(telx).view(-1)
    lyhat, elyhat = tlyhat.data.cpu().numpy(), telyhat.data.cpu().numpy()
    
    R = compute_R(lx, lyhat)
    
    print "C: %.2f || R: %.2f || ERM Loss: %s " % (loss[1], R, loss[0])

    plt.figure(figsize=(400 / 80., 300 / 80.))
    
    plt.scatter(x, y, s=30, c='r')
    textstrC = r'$C(\theta)=%.2f$' % (loss[1], )
    textstrR = r'$R(h_\theta)=%.2f$' % (R, )
                
    propsR = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    propsC = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.03, 0.05, textstrR, fontsize=18, transform=plt.gca().transAxes, verticalalignment='bottom', bbox=propsR)
    plt.text(0.50, 0.05, textstrC, fontsize=18, transform=plt.gca().transAxes, verticalalignment='bottom', bbox=propsC)

    plt.plot(elx, elyhat, color='blue')
    plt.xlabel(r'$$x$$')
    plt.ylabel(r'$$h_\theta(x)$$')
    plt.ylim(min(elyhat), max(elyhat))
    plt.xlim(min(elx), max(elx))
    plt.tight_layout()
    plt.savefig("images/fig_N%s_L%s_m%s_D%s_epoch%s.png"%(num_points, num_layers, num_hidden, lmbda, epoch), dpi = 200)
    plt.close()

def compute_R(lx, ly):
    lx = lx.reshape(-1)
    slopes = [(ly[i+1] - ly[i])/(lx[i+1] - lx[i]) for i in range(ly.shape[0]-1)]
    absdiffs = [abs(slopes[i+1] - slopes[i]) for i in range(len(slopes)-1)]
    R = sum(absdiffs)
    return R
