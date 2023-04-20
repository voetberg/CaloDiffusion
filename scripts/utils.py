import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from CaloChallenge.code.XMLHandler import *
from consts import *

#use tqdm if local, skip if batch job
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable

def split_data_np(data, frac=0.8):
    np.random.shuffle(data)
    split = int(frac * data.shape[0])
    train_data =data[:split]
    test_data = data[split:]
    return train_data,test_data


def create_R_Z_image(device, scaled = False, shape = (1,45,16,9)):

    if(shape[-1] == 30): #dataset 1, photons
        r_bins =  [ 0.,  2.,  4.,  5.,  6.,  8., 10., 12., 15., 20., 25., 30., 40., 50., 60., 70., 80., 90.,  100.,  
                      120., 130.,  150.,  160.,  200.,  250.,  300.,  350.,  400.,  600., 1000., 2000.]
    elif(shape[-1] == 23): #dataset 1, pions
        r_bins = [0.00, 1.00, 4.00, 5.00, 7.00, 10.00, 15.00, 20.00, 30.00, 50.00, 80.00, 90.00, 100.00, 
                130.00, 150.00, 160.00, 200.00, 250.00, 300.00, 350.00, 400.00, 600.00, 1000.00, 2000.00]
    elif(shape[-1] == 9): #dataset 2
        r_bins = [0,4.65,9.3,13.95,18.6,23.25,27.9,32.55,37.2,41.85]
    else:#dataset 3
        r_bins = [0,2.325,4.65,6.975,9.3,11.625,13.95,16.275,18.6,20.925,23.25,25.575,27.9,30.225,32.55,34.875,37.2,39.525,41.85]

    r_avgs = [(r_bins[i] + r_bins[i+1]) / 2.0 for i in range(len(r_bins) -1) ]
    assert(len(r_avgs) == shape[-1])
    Z_image = torch.zeros(shape, device=device)
    R_image = torch.zeros(shape, device=device)
    for z in range(shape[1]):
        Z_image[:,z,:,:] = z

    for r in range(shape[-1]):
        R_image[:,:,:,r] = r_avgs[r] 
    if(scaled):
        r_max = r_avgs[-1]
        z_max = shape[1]
        Z_image /= z_max
        R_image /= r_max
    return R_image, Z_image


def split_data(data,nevts,frac=0.8):
    data = data.shuffle(nevts)
    train_data = data.take(int(frac*nevts)).repeat()
    test_data = data.skip(int(frac*nevts)).repeat()
    return train_data,test_data

line_style = {
    'Geant4':'dotted',
    'CaloScore: VP':'-',
    'CaloScore: VE':'-',
    'CaloScore: subVP':'-',

    'VP 50 steps':'v',
    'VE 50 steps':'v',
    'subVP 50 steps':'v',

    'VP 500 steps':'^',
    'VE 500 steps':'^',
    'subVP 500 steps':'^',

    'VP r=0.0':'v',
    'VE r=0.0':'v',
    'subVP r=0.0':'v',

    'VP r=0.3':'^',
    'VE r=0.3':'^',
    'subVP r=0.3':'^',

    'Autoencoder' : '-',
    'Diffu' : '-',
    'Avg Shower' : '-',
    'LatentDiffu' : '-',

}

colors = {
    'Geant4':'black',
    'CaloScore: VP':'#7570b3',
    'CaloScore: VE':'#d95f02',
    'CaloScore: subVP':'#1b9e77',

    'VP 50 steps':'#e7298a',
    'VE 50 steps':'#e7298a',
    'subVP 50 steps':'#e7298a',

    'VP 500 steps':'#e7298a',
    'VE 500 steps':'#e7298a',
    'subVP 500 steps':'#e7298a',

    'VP r=0.0':'#e7298a',
    'VE r=0.0':'#e7298a',
    'subVP r=0.0':'#e7298a',

    'VP r=0.3':'#e7298a',
    'VE r=0.3':'#e7298a',
    'subVP r=0.3':'#e7298a',
    
    'Autoencoder': 'purple',
    'Diffu': 'blue',
    'Avg Shower': 'blue',
    'LatentDiffu': 'green',

}

name_translate={

    'AE' : "Autoencoder",
    'Diffu' : "Diffu",
    'LatentDiffu' : "LatentDiffu",
    'Avg' : "Avg Shower",


    'VPSDE':'CaloScore: VP',
    'VESDE':'CaloScore: VE',
    'subVPSDE':'CaloScore: subVP',

    '50_VPSDE':'VP 50 steps',
    '50_VESDE':'VE 50 steps',
    '50_subVPSDE':'subVP 50 steps',

    '500_VPSDE':'VP 500 steps',
    '500_VESDE':'VE 500 steps',
    '500_subVPSDE':'subVP 500 steps',

    '0p0_VPSDE':'VP r=0.0',
    '0p0_VESDE':'VE r=0.0',
    '0p0_subVPSDE':'subVP r=0.0',

    '0p3_VPSDE':'VP r=0.3',
    '0p3_VESDE':'VE r=0.3',
    '0p3_subVPSDE':'subVP r=0.3',

    
    }

def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    #import mplhep as hep
    #hep.set_style(hep.style.CMS)
    #hep.style.use("CMS") 

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            eps = 1e-8
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0) + eps)
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.2f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')

def make_histogram(entries, labels, colors, xaxis_label="", title ="", num_bins = 10, logy = False, normalize = False, stacked = False, h_type = 'step', 
        h_range = None, fontsize = 16, fname="", yaxis_label = "", ymax = -1):
    alpha = 1.
    if(stacked): 
        h_type = 'barstacked'
        alpha = 0.2
    fig_size = (8,6)
    fig = plt.figure(figsize=fig_size)
    ns, bins, patches = plt.hist(entries, bins=num_bins, range=h_range, color=colors, alpha=alpha,label=labels, density = normalize, histtype=h_type)
    plt.xlabel(xaxis_label, fontsize =fontsize)
    plt.tick_params(axis='x', labelsize=fontsize)

    if(logy): plt.yscale('log')
    elif(ymax > 0):
        plt.ylim([0,ymax])
    else:
        ymax = 1.3 * np.amax(ns)
        plt.ylim([0,ymax])

    if(yaxis_label != ""):
        plt.ylabel(yaxis_label, fontsize=fontsize)
        plt.tick_params(axis='y', labelsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(loc='upper right', fontsize = fontsize)
    if(fname != ""): 
        plt.savefig(fname)
        print("saving fig %s" %fname)
    #else: plt.show(block=False)
    return fig


def HistRoutine(feed_dict,xlabel='',ylabel='',reference_name='Geant4',logy=False,binning=None,label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),10)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    
    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            dist,_ = np.histogram(feed_dict[plot],bins=binning,density=True)
            ax0.plot(xaxis,dist,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3,label=plot)
            #dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,marker=line_style[plot],color=colors[plot],density=True,histtype="step")
        else:
            dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=plot,linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
            
        if reference_name!=plot:
            eps = 1e-8
            ratio = 100*np.divide(dist - reference_hist,reference_hist + eps)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(xaxis,ratio,color=colors[plot],marker=line_style[plot],ms=10,lw=0,markeredgewidth=3)
            else:
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0)
        
    ax0.legend(loc=label_loc,fontsize=16,ncol=1)        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0) 

    if logy:
        ax0.set_yscale('log')
    
    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-50,50])

    return fig,ax0






def DataLoader(file_name,shape,emax,emin, nevts=-1, max_deposit=2, logE=True, showerMap = 'log-norm', nholdout = 0, from_end = False, dataset_num = 2, fully_connected = False):

    start = 0
    with h5.File(file_name,"r") as h5f:
        #holdout events for testing
        if(nevts == -1 and nholdout > 0): nevts = -(nholdout)
        end = int(nevts)
        if(from_end):
            start = -int(nevts)
            end = None
        e = h5f['incident_energies'][start:end].astype(np.float32)/1000.0
        shower = h5f['showers'][start:end].astype(np.float32)/1000.0

        
    e = np.reshape(e,(-1,1))
    shower = np.reshape(shower,(shower.shape[0],-1))
    shower = shower/(max_deposit*e)


    shower_preprocessed = preprocess_shower(shower, shape, showerMap, dataset_num = dataset_num, fully_connected = fully_connected)

    if logE:        
        E_preprocessed = np.log10(e/emin)/np.log10(emax/emin)
    else:
        E_preprocessed = (e-emin)/(emax-emin)

    return shower_preprocessed, E_preprocessed 
    
def preprocess_shower(shower, shape, showerMap = 'log-norm', dataset_num = 2, fully_connected = False):

    if(dataset_num > 1): 
        shower = shower.reshape(shape)
    elif(not fully_connected):
        if(dataset_num == 1): 
            binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", binning_file)
        else: 
            binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", binning_file)

        g = GeomConverter(bins)
        shower = g.convert(g.reshape(shower))



    if(dataset_num > 3 or dataset_num <0 ): 
        print("Invalid dataset %i!" % dataset_num)
        exit(1)

    if(fully_connected and dataset_num <= 1): dataset_num +=10 

    print('dset', dataset_num)

    c = dataset_params[dataset_num]


    if('logit' in showerMap):
        alpha = 1e-6
        x = alpha + (1 - 2*alpha)*shower
        shower = np.ma.log(x/(1-x)).filled(0)    

        if('norm' in showerMap): shower = (shower - c['logit_mean']) / c['logit_std']
        elif('scaled' in showerMap): shower = 2.0 * (shower - c['logit_min']) / (c['logit_max'] - c['logit_min']) - 1.0

    elif('log' in showerMap):
        eps = 1e-8
        shower = np.ma.log(shower).filled(c['log_min'])
        if('norm' in showerMap): shower = (shower - c['log_mean']) / c['log_std']
        elif('scaled' in showerMap):  shower = 2.0 * (shower - c['log_min']) / (c['log_max'] - c['log_min']) - 1.0

    elif('sqrt' in showerMap):
        shower = np.sqrt(shower)
        if('norm' in showerMap): shower = (shower - c['sqrt_mean']) / c['sqrt_std']
        #Range naturally from 0 to 1, change to be from -1 to 1
        elif('scaled' in showerMap): shower  = (shower * 2.0) - 1.0

    return shower

        

def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))


def ReverseNorm(voxels,e,shape,emax,emin,max_deposit=2,logE=True, showerMap ='log', dataset_num = 2, fully_connected = False):
    '''Revert the transformations applied to the training set'''

    if(dataset_num > 3 or dataset_num <0 ): 
        print("Invalid dataset %i!" % dataset_num)
        exit(1)
    if(fully_connected and dataset_num <= 1): dataset_num +=10 
    print('dset', dataset_num)
    c = dataset_params[dataset_num]


    #shape=voxels.shape
    alpha = 1e-6
    if logE:
        energy = emin*(emax/emin)**e
    else:
        energy = emin + (emax-emin)*e
        
    if('logit' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c['logit_std']) + c['logit_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c['logit_max'] - c['logit_min']) + c['logit_min']

        #avoid overflows
        #voxels = np.minimum(voxels, np.log(max_deposit/(1-max_deposit)))

        exp = np.exp(voxels)    
        x = exp/(1+exp)
        data = (x-alpha)/(1 - 2*alpha)

    elif('log' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c['log_std']) + c['log_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0) * 0.5 * (c['log_max'] - c['log_min']) + c['log_min']

        voxels = np.minimum(voxels, np.log(max_deposit))


        data = np.exp(voxels)

    elif('sqrt' in showerMap):
        if('norm' in showerMap): voxels = (voxels * c['sqrt_std']) + c['sqrt_mean']
        elif('scaled' in showerMap): voxels = (voxels + 1.0)/2.0
        data = np.square(voxels)




    if(dataset_num > 1 or fully_connected): 
        print("reshape")
        data = data.reshape(voxels.shape[0],-1)*max_deposit*energy.reshape(-1,1)
    else:
        if(dataset_num == 1): 
            binning_file = "../CaloChallenge/code/binning_dataset_1_photons.xml"
            bins = XMLHandler("photon", binning_file)
        else: 
            binning_file = "../CaloChallenge/code/binning_dataset_1_pions.xml"
            bins = XMLHandler("pion", binning_file)
        g = GeomConverter(bins)
        data = np.squeeze(data)
        data = g.unreshape(g.unconvert(data))*max_deposit*energy.reshape(-1,1)
    
    return data,energy
    

def polar_to_cart(polar_data,nr=9,nalpha=16,nx=12,ny=12):
    cart_img = np.zeros((nx,ny))
    ntotal = 0
    nfilled = 0

    if not hasattr(polar_to_cart, "x_interval"):
        polar_to_cart.x_interval = np.linspace(-1,1,nx)
        polar_to_cart.y_interval = np.linspace(-1,1,ny)
        polar_to_cart.alpha_pos = np.linspace(1e-5,1,nalpha)
        polar_to_cart.r_pos = np.linspace(1e-5,1,nr)

    for alpha in range(nalpha):
        for r in range(nr):
            if(polar_data[alpha,r] > 0):
                x = polar_to_cart.r_pos[r] * np.cos(polar_to_cart.alpha_pos[alpha]*np.pi*2)
                y = polar_to_cart.r_pos[r] * np.sin(polar_to_cart.alpha_pos[alpha]*np.pi*2)
                binx = np.argmax(polar_to_cart.x_interval>x)
                biny = np.argmax(polar_to_cart.y_interval>y)
                ntotal+=1
                if cart_img[binx,biny] >0:
                    nfilled+=1
                cart_img[binx,biny]+=polar_data[alpha,r]
    return cart_img

class GeomConverter:
    "Convert irregular geometry to regular one (ala CaloChallenge Dataset 1)"
    def __init__(self, bins = None, all_r_edges = None, lay_r_edges = None, alpha_out = 1, lay_alphas = None):

        self.layer_boundaries = []
        self.bins = None

        #init from binning
        if(bins is not None):
            

            self.layer_boundaries = np.unique(bins.GetBinEdges())
            rel_layers = bins.GetRelevantLayers()
            lay_alphas = [len(bins.alphaListPerLayer[idx][0]) for idx, redge in enumerate(bins.r_edges) if len(redge) > 1]
            alpha_out = np.amax(lay_alphas)


            all_r_edges = []

            lay_r_edges = [bins.r_edges[l] for l in rel_layers]
            for ilay in range(len(lay_r_edges)):
                for r_edge in lay_r_edges[ilay]:
                    all_r_edges.append(r_edge)
            all_r_edges = torch.unique(torch.FloatTensor(all_r_edges))

        self.all_r_edges = all_r_edges
        self.lay_r_edges = lay_r_edges
        self.alpha_out = alpha_out
        self.lay_alphas = lay_alphas
        self.num_layers = len(self.lay_r_edges)


        self.all_r_areas = (all_r_edges[1:]**2 - all_r_edges[:-1]**2)
        self.dim_r_out = len(all_r_edges) - 1
        self.weight_mats = []
        for ilay in range(len(lay_r_edges)):
            dim_in = len(lay_r_edges[ilay]) - 1
            weight_mat = torch.zeros((self.dim_r_out, dim_in))
            for ir in range(dim_in):
                o_idx_start = torch.nonzero(self.all_r_edges == self.lay_r_edges[ilay][ir])[0][0]
                o_idx_stop = torch.nonzero(self.all_r_edges == self.lay_r_edges[ilay][ir + 1])[0][0]

                split_idxs = list(range(o_idx_start, o_idx_stop))
                orig_area = (self.lay_r_edges[ilay][ir+1]**2 - self.lay_r_edges[ilay][ir]**2)

                #split proportional to bin area
                weight_mat[split_idxs, ir] = self.all_r_areas[split_idxs]/orig_area

            self.weight_mats.append(weight_mat)



    def reshape(self, raw_shower):
        #convert to jagged array each of shape (N_shower, N_alpha, N_R)
        shower_reshape = []
        for idx in range(len(self.layer_boundaries)-1):
            data_reshaped = raw_shower[:,self.layer_boundaries[idx]:self.layer_boundaries[idx+1]].reshape(raw_shower.shape[0], int(self.lay_alphas[idx]), -1)
            shower_reshape.append(data_reshaped)
        return shower_reshape

    def unreshape(self, raw_shower):
        #convert jagged back to original flat format
        n_show = raw_shower[0].shape[0]
        out = np.zeros((n_show, self.layer_boundaries[-1]))
        for idx in range(len(self.layer_boundaries)-1):
            out[:, self.layer_boundaries[idx]:self.layer_boundaries[idx+1]] = raw_shower[idx].reshape(n_show, -1)
        return out


    def convert(self, d):
        out = torch.zeros((len(d[0]), self.num_layers, self.alpha_out, self.dim_r_out))
        for i in range(len(d)):
            o = torch.einsum( '...ij,...j->...i', self.weight_mats[i], torch.FloatTensor(d[i]))
            if(self.lay_alphas is not None):
                if(self.lay_alphas[i]  == 1):
                    #distribute evenly in phi
                    o = torch.repeat_interleave(o, self.alpha_out, dim = -2)/self.alpha_out
                elif(self.lay_alphas[i]  != self.alpha_out):
                    print("Num alpha bins for layer %i is %i. Don't know how to handle" % (i, self.lay_alphas[i]))
                    exit(1)
            out[:,i] = o
        return out


    def unconvert(self, d):
        out = []
        for i in range(self.num_layers):
            o = torch.einsum( '...ij,...j->...i', torch.linalg.pinv(self.weight_mats[i]), torch.FloatTensor(d[:,i]))
            if(self.lay_alphas is not None):
                if(self.lay_alphas[i]  == 1):
                    #Only works for converting 1 alpha bin into multiple, ok for dataset1 but maybe should generalize
                    o = torch.sum(o, dim = -2, keepdim = True)
                elif(self.lay_alphas[i]  != self.alpha_out):
                    print("Num alpha bins for layer %i is %i. Don't know how to handle" % (i, self.lay_alphas[i]))
                    exit(1)
            out.append(o)
        return out


class EarlyStopper:
    def __init__(self, patience=1, mode = 'loss', min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.mode = mode

    def early_stop(self, var):
        if(self.mode == 'val_loss'):
            validation_loss = var
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False
        elif(self.mode == 'diff'):
            if(var < 0):
                self.counter = 0
            else:
                self.counter += 1
                if( self.counter >= self.patience):
                    return True
            return False



def SetFig(xlabel,ylabel):
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1) 
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_ticks_position('both')
    ax0.xaxis.set_ticks_position('both')
    ax0.tick_params(direction="in",which="both")    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel(ylabel,fontsize=20)

    ax0.minorticks_on()
    return fig, ax0


if __name__ == "__main__":
    #Preprocessing of the input files: conversion to cartesian coordinates + zero-padded mask generation
    file_path = '/wclustre/cms/denoise/CaloChallenge/dataset_2_2.hdf5'
    #with h5.File(file_path,"r") as h5f:
    #    e = h5f['incident_energies'][:]
    #    showers = h5f['showers'][:]
    ##shape = [-1,45,50,18,1]
    ##nx=32
    ##ny=32
    #
    #shape = [-1,45,16,9,1]

    #nx = 12
    #ny = 12

    #showers = showers.reshape((-1,shape[2],shape[3]))
    #cart_data = []
    #for ish,shower in enumerate(showers):
    #    if ish%(10000)==0:print(ish)
    #    cart_data.append(polar_to_cart(shower,nr=shape[3],nalpha=shape[2],nx=nx,ny=ny))

    #cart_data = np.reshape(cart_data,(-1,45,nx,ny))
    #with h5.File('/wclustre/cms/denoise/CaloChallenge/dataset_2_2_cart.hdf5',"w") as h5f:
    #    dset = h5f.create_dataset("showers", data=cart_data)
    #    dset = h5f.create_dataset("incident_energies", data=e)


    file_path = '/wclustre/cms/denoise/CaloChallenge/dataset_2_1_cart.hdf5'
    ##file_path='/wclustre/cms/denoise/CaloChallenge/dataset_1_photons_1.hdf5'
    with h5.File(file_path,"r") as h5f:
        showers = h5f['showers'][:]
        energies = h5f['incident_energies'][:]
    mask = np.sum(showers,0)==0
    mask_file = file_path.replace('.hdf5','_mask.hdf5')
    print("Creating mask file %s " % mask_file)
    with h5.File(mask_file,"w") as h5f:
        dset = h5f.create_dataset("mask", data=mask)
    
