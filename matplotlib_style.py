#Define style parameters for any graphs
#Based on https://github.com/kdungs/lhcb-matplotlibrc/blob/master/matplotlibrc

def matplotlib_style(**kwargs):
    
    matplotlib_style = {
    "axes.labelsize" : 46,
    "axes.linewidth" : 2,
    
    "figure.figsize" : (8, 8),
    "figure.dpi" : 100,
    
    "text.usetex" : True,
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"""
    \usepackage[utf8]{inputenc}
    \DeclareUnicodeCharacter{2212}{-}  % Handles minus sign
    \usepackage{amsmath}  % For math rendering
    \usepackage{siunitx}  % For units
    \usepackage{sansmath} % Use sans-serif math font
    \sansmath  % Activate sans-serif math mode
    """,
    "font.family" : "sans-serif",
    
    "font.size" : 35,
    "font.weight" : 400,
    
    "legend.frameon" : False,
    "legend.handletextpad" : 0.3,
    "legend.numpoints" : 1,
    "legend.labelspacing" : 0.2,
    "legend.fontsize" : 35,
    
    "lines.linewidth" : 2,
    "lines.markeredgewidth" : 0,
    "lines.markersize" : 8,
    
    "savefig.bbox" : "tight",
    "savefig.pad_inches" : 0.1,
 

    "xtick.direction" : 'out',
    "xtick.major.size" : 14,
    "xtick.minor.size" : 7,
    "xtick.major.width" : 1.5,
    "xtick.minor.width" : 1.5,
    "xtick.major.pad" : 10,
    "xtick.minor.pad" : 10,
    "xtick.labelsize" : 35,
    "xtick.minor.visible": True,
    
    "ytick.direction" : 'out',
    "ytick.major.size" : 14,
    "ytick.minor.size" : 7,
    "ytick.major.width" : 1.5,
    "ytick.minor.width" : 1.5,
    "ytick.major.pad" : 10,
    "ytick.minor.pad" : 10,
    "ytick.labelsize" : 35,
    "ytick.minor.visible": True,
    }
    
    for key, value in kwargs.items():
        matplotlib_style[key] = value
        
    return matplotlib_style
    
if __name__ == '__main__':
    #Example graph with this style    
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.rcParams.update(matplotlib_style(**{"text.usetex":False}))
    
    x = np.linspace(-5,5,num=100)
    y = np.power(x, 3)
    
    fig, ax = plt.subplots()
    ax.set_axisbelow(False)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.plot(x,y)
    
    plt.show()