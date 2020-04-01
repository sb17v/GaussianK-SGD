import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CDF PDF Plotting Script")
    parser.add_argument('--filename', nargs='+', default='weights.pickle')
    parser.add_argument('--legend', nargs='+', default='weight')
    args = parser.parse_args()
    filename = args.filename
    legends = args.legend
    counter = 1
    fig = plt.figure(figsize=(10,6))
    for f in filename:
        print(f)
        x = np.load(f, allow_pickle=True)
        print("Name: %s Min: %s Max: %s" %(f.split('/')[-1], min(x), max(x)))
        #norm_cdf = scipy.stats.norm.cdf(x) # calculate the cdf - also discrete
        #norm_pdf = scipy.stats.norm.pdf(x)
        # plot the cdf
        #sns.lineplot(x=x, y=norm_cdf)
        sns.distplot(x, kde = True, kde_kws = {"shade": True, "legend": True}, rug = False, hist = False, axlabel="Gradients", label=legends[counter-1])
        #g.set_yscale("log")
        #sns.distplot(x, kde = False, rug = True, norm_hist = True, hist_kws={"histtype": "step", "linewidth": 1, "alpha": 1}, rug_kws={"height": str(counter * 0.05)}, axlabel="Gradients" )
        #sns.distplot(x, rug=True, rug_kws={"color": "g"}, kde_kws={"color": "k", "lw": 3, "label": "KDE"}, hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})
        # plot the pdf
        #sns.lineplot(x=x, y=norm_pdf)
        counter += 1
    #fig.legend(labels=legends)
    #plt.savefig('plot.pdf')
    plt.savefig('plot.png')
    
