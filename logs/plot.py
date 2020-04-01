# -*- coding: utf-8 -*-


import time
import datetime
import itertools
import argparse
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import utils as u

markers=[None]
colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'purple', 'olive']
markeriter = itertools.cycle(markers)
coloriter = itertools.cycle(colors)

OUTPUTPATH='/home/sbhatt/codebase/GaussianK-SGD/logs/'
LOGPATH='/home/sbhatt/codebase/GaussianK-SGD/logs/'

EPOCH = True
FONTSIZE=16
LEGENDSIZE=8

fig, ax = plt.subplots(1,1,figsize=(5,3.5))
ax2 = None

STANDARD_TITLES = {
        'resnet20': 'ResNet-20',
        'vgg16': 'VGG-16',
        'alexnet': 'AlexNet',
        'resnet50': 'ResNet-50',
        'lstmptb': 'LSTM-PTB',
        'lstman4': 'LSTM-an4'
        }

def get_real_title(title):
    return STANDARD_TITLES.get(title, title)

def seconds_between_datetimestring(a, b):
    a = datetime.datetime.strptime(a, '%Y-%m-%d %H:%M:%S')
    b = datetime.datetime.strptime(b, '%Y-%m-%d %H:%M:%S')
    delta = b - a 
    return delta.days*86400+delta.seconds
    
sbd = seconds_between_datetimestring

def get_acc_or_loss(line, isacc=False):
    valid = line.find('top-5 acc: ') > 0 if isacc else line.find('val loss: ') > 0
    if line.find('Epoch') > 0 and valid: 
        items = line.split(' ')
        loss = float(items[-1])
        t = line.split(' I')[0].split(',')[0]
        t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        return loss, t
    return None, None

def read_data_from_log(logfile, plot_type):
    f = open(logfile)

    accuracies = []
    losses = []
    times = []
    average_delays = []
    lrs = []

    i = 0
    time0 = None 
    max_epochs = 500
    counter = 0

    for line in f.readlines():
        if plot_type == 'accuracy' or plot_type == 'latency':                  #Finding Accuracy
            valid = line.find('top-5 acc: ') > 0
        if plot_type == 'loss' or plot_type == 'latency':                      #Finding Loss
            valid = line.find('val loss: ') > 0
        if plot_type == 'latency':                   #Calculating Time
            if line.find('Epoch') > 0 and valid:    
                t = line.split(' I')[0].split(',')[0]
                t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                if not time0:
                    time0 = t
        if plot_type == 'lr':                        #Calculating lr changes
            if line.find('lr: ') > 0:
                try:
                    lr = float(line.split(',')[-2].split('lr: ')[-1])
                    lrs.append(lr)
                except:
                    pass 

        if line.find('average delay: ') > 0:    #Calculating average delay
            delay = int(line.split(':')[-1])
            average_delays.append(delay)

        if plot_type == 'accuracy' or plot_type == 'latency':
            accuracy, t = get_acc_or_loss(line, True)
            if accuracy and t:
                counter +=1
                accuracies.append(accuracy)
                times.append(t)
            if counter > max_epochs:
                break
        
        if plot_type == 'loss' or plot_type == 'latency':
            loss, t = get_acc_or_loss(line, False)
            if loss and t:
                counter +=1
                losses.append(loss)
                times.append(t)
            if counter > max_epochs:
                break
        
        f.close()
   
    return accuracies, losses, times, average_delays, lrs

def read_norm_from_log(logfile):
    f = open(logfile)
    means = []
    stds = []
    for line in f.readlines():
        if line.find('gtopk-dense norm mean') > 0:
            items = line.split(',')
            mean = float(items[-2].split(':')[-1])
            std = float(items[--1].split(':')[-1])
            means.append(mean)
            stds.append(std)
    print('means: ', means)
    print('stds: ', stds)
    return means, stds

def plot_data(logfile, plot_type, label, title='ResNet-20'):
    accuracies, losses, times, average_delays, lrs = read_data_from_log(logfile, plot_type)

    data = []

    if plot_type == 'accuracy':
        data = accuracies
    elif plot_type == 'loss':
        data = losses
    elif plot_type == 'latency':
        average_interval = 10
        if len(times) > 0:
            for i in range(1, len(times)):
                delta = times[i]- times[i-1]
                data.append(delta.days*86400+delta.seconds)
    elif plot_type == 'lr':
        data = lrs
    else:
        print('Plot type not defined till now')
        exit()


    if logfile.find('resnet50') > 0 or logfile.find('alexnet') > 0:
        losses = losses[0:45]
        accuracies = accuracies[0:45]
    
    print('Data: ', data)

    norm_means, norm_stds = read_norm_from_log(logfile)

    if len(average_delays) > 0:
        delay = int(np.mean(average_delays))
    else:
        delay = 0

    if delay > 0:
        label = label + ' (delay=%d)' % delay
    
    ax.set_ylabel(plot_type.capitalize())
    ax.set_title(get_real_title(title))

    marker = next(markeriter)
    color = next(coloriter)
    ax.plot(list(range(0,len(data))), data, label=label, marker=marker, markerfacecolor='none', color=color)
    
    from matplotlib.ticker import MaxNLocator,LinearLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    #ax.tick_params(labelsize=8)

    #Only special cases:
    if False and len(norm_means) > 0:
        global ax2
        if ax2 is None:
            ax2 = ax.twinx()
            ax2.set_ylabel('L2-Norm of : gTopK-Dense')
        ax2.plot(norm_means, label=label+' norms', color=color)

    ax.set_xlabel('Epoch')

    ax.grid(linestyle=':')
    if len(lrs) > 0:
        lr_indexes = [0]
        lr = lrs[0]
        for i in range(len(lrs)):
            clr = lrs[i]
            if lr != clr:
                lr_indexes.append(i)
                lr = clr

    u.update_fontsize(ax, FONTSIZE)



def plot_graph(network, plot_type, logfile_names, legends):
    for l in range(len(logfile_names)):
        logfile = LOGPATH + '/' + logfile_names[l]
        legend = legends[l]

        plot_data(logfile, plot_type, legend, network)

    ax.set_xlim(xmin=-1)
    ax.legend(fontsize=LEGENDSIZE)
    plt.subplots_adjust(bottom=0.18, left=0.2, right=0.96, top=0.9)
    plt.savefig('%s/%s_%s.pdf' % (OUTPUTPATH, network, plot_type), dpi=400)
    plt.savefig('%s/%s_%s.png' % (OUTPUTPATH, network, plot_type), dpi=400)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plotting Script")
    parser.add_argument('--network', type=str, default='resnet20', required=True)
    parser.add_argument('--plot-type', type=str, default='acc', required=True)
    parser.add_argument('--logfile-names', nargs='+', required=True)
    parser.add_argument('--legends', nargs='+', required=True)
    
    args = parser.parse_args()
    print(args.logfile_names)
    plot_graph(args.network, args.plot_type, args.logfile_names, args.legends)
