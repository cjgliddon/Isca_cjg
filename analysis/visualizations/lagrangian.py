import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_err_scatter(err_dict, savename="err_scatter.gif"):
    """
    Generates an animation of the growth in position and intensity errors of
    feature trajectories in ensembles of predictability experiments. Position
    errors are represented by scatterplot points in Cartesian space (origin =
    zero error); intensity errors are denoted by the color of scatterplot
    points (default: red = positive error, blue = negative error).

    Arguments
    ---------
    err_dict : dict
        Dictionary whose keys are floats representing times after track
        genesis, and whose values are three-element lists containing
        lists (or arrays) of the errors in x, y, and intensity between all
        matching ensemble and control track pairs at the corresponding time.

    savename : str, optional
        Path (either relative or absolute) to which animation should be saved.

    Returns
    -------
    anim : FuncAnimation
    """
    times = list(err_dict.keys())
    
    # Determine scatterplot axis and colorbar ranges
    dx_max = max([max(np.abs(err_dict[t][0])) for t in times])
    dy_max = max([max(np.abs(err_dict[t][1])) for t in times])
    dint_max = max([max(np.abs(err_dict[t][2])) for t in times])

    fig = plt.figure()
    ax = plt.axes()
    
    # Create initial scatter plot and colorbar (outside update loop)
    t_init = times[0]
    xp_init, yp_init, intp_init = err_dict[t_init]
    scatter = ax.scatter(xp_init, yp_init, c=intp_init, cmap='RdBu_r', 
                        vmin=-dint_max, vmax=dint_max, s=50)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("$Δ$intensity")

    def update(idx):
        ax.cla()
        t = times[idx]
        xp, yp, intp = err_dict[t]
        
        # Update scatter plot
        scatter = ax.scatter(xp, yp, c=intp, cmap='RdBu_r', 
                            vmin=-dint_max, vmax=dint_max, s=40,
                            edgecolors='k')
        
        ax.set_xlabel("$Δx$ (km)")
        ax.set_ylabel("$Δy$ (km)")
        ax.set_title(f"time after genesis = {t} days")
        ax.set_xlim(-dx_max, dx_max)
        ax.set_ylim(-dy_max, dy_max)

    anim = FuncAnimation(fig, update, frames=len(times))
    anim.save(savename)
    
    return anim