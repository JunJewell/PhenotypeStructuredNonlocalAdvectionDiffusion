"""
Phenotype x Space plotting utilities.

Provides:
- PhenotypePlotter: heatmaps & animations of u(x,theta,t) with x on horizontal axis
  and theta on vertical axis.

Expected inputs:
- solution_y : 2D ndarray-like with shape compatible with (n_theta * mesh_points_x, n_time)
               OR shape (n_theta, mesh_points_x, n_time) OR (mesh_points_x, n_theta, n_time).
               The class will attempt to reshape robustly.
- solution_t : 1D array of time points
- L_x        : length of physical domain in x
- mesh_points_x : number of spatial grid points in x
- n_theta    : number of phenotype bins (theta)
- L_theta    : phenotype domain length (default 1.0)
- plotting options: folder_tag, file_tag, save, cmap, rasterize, num_ticks.

"""

AX_LEFT = 0.22 
AX_BOTTOM = 0.22
AX_WIDTH = 0.52
AX_HEIGHT = 0.58

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import os

import matplotlib
from matplotlib_style import matplotlib_style

from matplotlib.backends.backend_agg import FigureCanvasAgg


def theta_grid(ntheta):
    """Cell-centred theta grid on [0,1]."""
    if ntheta <= 0:
        raise ValueError("ntheta must be > 0")
    return (np.arange(ntheta) + 0.5) / ntheta

class PhenotypePlotter:
    def __init__(
        self,
        solution_y, solution_t, L_x, mesh_points_x, n_theta, folder_tag="results",
        file_tag="phenotype", L_theta=1.0, save=True, cmap="inferno", rasterize_heatmap=True,
        num_heatmap_ticks=6,
    ):
        """
        Parameters
        ----------
        solution_y : array-like
            Solution data saved by run script. Accepted shapes:
              - (n_theta * mesh_points_x, n_time)
              - (n_theta, mesh_points_x, n_time)
              - (mesh_points_x, n_theta, n_time)
        solution_t : array-like, shape (n_time,)
        L_x : float
            domain length in x (physical)
        mesh_points_x : int
        n_theta : int
        folder_tag, file_tag : strings for saving outputs
        L_theta : float (default 1.0)
        save : bool
        cmap : string (matplotlib colormap)
        rasterize_heatmap : bool (whether pcolormesh is rasterized)
        num_heatmap_ticks : int (axis tick control)
        """
        
        plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
        matplotlib.use('pgf')
        
        self.L_x = float(L_x)
        self.mesh_points_x = int(mesh_points_x)
        self.n_theta = int(n_theta)
        self.L_theta = float(L_theta)
        self.folder_tag = folder_tag
        self.file_tag = file_tag
        self.save = bool(save)
        self.cmap = cmap
        self.rasterize_heatmap = bool(rasterize_heatmap)
        self.num_heatmap_ticks = int(num_heatmap_ticks)

        # time array
        self.solution_t = np.asarray(solution_t).flatten()
        if self.solution_t.ndim != 1:
            raise ValueError("solution_t must be 1D array of time points")
        self.n_time = self.solution_t.size

        # theta and x grids
        self.theta = theta_grid(self.n_theta) * (self.L_theta)  # cell centres in [0,L_theta]
        self.dtheta = self.L_theta / self.n_theta
        self.x_points = np.linspace(0.0, self.L_x, self.mesh_points_x, endpoint=False)

        # Convert solution_y into canonical shape (n_theta, mesh_points_x, n_time)
        arr = np.asarray(solution_y)
        # If 2D and shape (rows, n_time) and rows == n_theta * mesh_points_x:
        if arr.ndim == 2 and arr.shape[0] == self.n_theta * self.mesh_points_x and arr.shape[1] == self.n_time:
            self.u = arr.reshape(self.n_theta, self.mesh_points_x, self.n_time)
        # If already given in (n_theta, mesh_points_x, n_time)
        elif arr.ndim == 3 and arr.shape[0] == self.n_theta and arr.shape[1] == self.mesh_points_x and arr.shape[2] == self.n_time:
            self.u = arr.copy()
        # If given in (mesh_points_x, n_theta, n_time), transpose axes
        elif arr.ndim == 3 and arr.shape[0] == self.mesh_points_x and arr.shape[1] == self.n_theta and arr.shape[2] == self.n_time:
            self.u = np.transpose(arr, (1, 0, 2))
        else:
            msg = ("Cannot interpret shape of solution_y. "
                   f"Received shape {arr.shape}. Expected one of: "
                   f"(n_theta*mesh_points_x, n_time), (n_theta, mesh_points_x, n_time), "
                   f"or (mesh_points_x, n_theta, n_time).")
            raise ValueError(msg)

        # Basic checks
        if np.any(self.u < -1e-12):
            print("WARNING: negative densities in input (tolerance 1e-12).")
        if np.any( np.sum(self.u, axis=0) * self.dtheta > 1.0 + 1e-12):
            print("WARNING: densities exceed 1 (tolerance 1e-12).")

        # Prepare folder if saving
        if self.save:
            os.makedirs(self.folder_tag, exist_ok=True)

    def _prepare_fig_axes(self, aspect_equal=False, figsize=(8, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        if aspect_equal:
            ax.set_aspect('equal', adjustable='box')
        return fig, ax

    def heatmap(self, time_index=-1, vmin=None, vmax=None, annotate_time=True, figsize=(8,6)):
        """
        Draw and (optionally) save a heatmap with x on horizontal axis and theta on vertical.
        time_index : int
        vmin, vmax : provide to fix color scale (otherwise auto)
        """
        if time_index < 0:
            time_index = self.n_time + time_index
        if not (0 <= time_index < self.n_time):
            raise IndexError("time_index out of range")

        fig, ax = self._prepare_fig_axes(figsize=figsize)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')

        # pcolormesh expects grid edges; produce edges for x and theta
        x_edges = np.linspace(0.0, self.L_x, self.mesh_points_x + 1)
        theta_edges = np.linspace(0.0, self.L_theta, self.n_theta + 1)

        # data must be shape (n_theta, mesh_points_x) where rows correspond to y-values (theta)
        data = self.u[:, :, time_index]

        pcm = ax.pcolormesh(x_edges, theta_edges, data, cmap=self.cmap, rasterized=self.rasterize_heatmap, vmin=vmin, vmax=vmax)
        

        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label(r"$u(x, \theta)$")
        
        cb.minorticks_off()

        ax.minorticks_off()
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\theta$")
        
        
        
        ax.xaxis.label.set_fontsize(100)
        ax.yaxis.label.set_fontsize(100)

        ax.set_yticks([0, 1 / 2, 1.])
 
        ax.set_box_aspect(0.9295303054)

        plt.tight_layout()
    
        
        if self.save:
            outpath = os.path.join(self.folder_tag, f"{self.file_tag}_heatmap_t{time_index}.pdf")
            fig.savefig(outpath, dpi=200)
            print(f"Saved heatmap to {outpath}")
        return fig, ax

    def final_snapshot(self, vmin=None, vmax=None, figsize=(8,6)):
        return self.heatmap(time_index=-1, vmin=vmin, vmax=vmax, figsize=figsize)
    
    def animate(self, fps=10, dpi=150, filename="phenotype_evolution.mp4",
                sample_rate=1, vmin=None, vmax=None,
                quick_start=True,):
        """
        Create and save an mp4 animation of the heatmap across time.
        sample_rate : take every `sample_rate`-th time point (1 -> every point)
        vmin,vmax : if provided, fix color scale; if None, autoscale per frame.
        """
        # don't use most of beginning as it is mainly for DMD
        if (quick_start and self.solution_t.size > 40*self.n_theta+1) :
            
            frame_indices_1 = list(range(0, 40*self.n_theta, 20))
            frame_indices_2 = list(range(40*self.n_theta, self.n_time, sample_rate))
            frame_indices = frame_indices_1 + frame_indices_2
        elif quick_start :
            frame_indices = list(range(0, self.n_time, 10*sample_rate))
        else:
            frame_indices = list(range(0, self.n_time, sample_rate))
        
        
        n_frames = len(frame_indices)
        if n_frames == 0:
            raise ValueError("No frames to render (check sample_rate and solution_t)")
    
        # Prepare figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])
        FigureCanvasAgg(fig)
        #div = make_axes_locatable(ax)
        cax = fig.add_axes([AX_LEFT + AX_WIDTH + 0.03,
                    AX_BOTTOM,
                    0.035,
                    AX_HEIGHT])
    
        x_edges = np.linspace(0.0, self.L_x, self.mesh_points_x + 1)
        theta_edges = np.linspace(0.0, self.L_theta, self.n_theta + 1)
    
        # initial data and initial clim
        data0 = self.u[:, :, frame_indices[0]]
        if vmin is None:
            frame_vmin = float(np.nanmin(data0))
        else:
            frame_vmin = vmin
        if vmax is None:
            frame_vmax = float(np.nanmax(data0))
        else:
            frame_vmax = vmax
    
        # create single QuadMesh mappable
        pcm = ax.pcolormesh(
            x_edges,
            theta_edges,
            data0,
            cmap=self.cmap,
            rasterized=self.rasterize_heatmap,
            vmin=frame_vmin,
            vmax=frame_vmax,
            #norm=norm,
            shading='auto'
        )
        cb = fig.colorbar(pcm, cax=cax)
        cb.set_label(r"$u(t,x,\theta)$")
    
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\theta$")
        ax.yaxis.set_major_locator(ticker.MaxNLocator(self.num_heatmap_ticks))
        
        ax.set_xticks([0, self.L_x/2, self.L_x])
    
        title = ax.set_title(f"time = {self.solution_t[frame_indices[0]]:.4g}")
    
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='PhenotypePlotter'), bitrate=4000)
        outpath = os.path.join(self.folder_tag, f"{self.file_tag}_{filename}")
    
        # Update function: update the QuadMesh array and clim, then update colorbar
        def update_frame(i):
            idx = frame_indices[i]
            data = self.u[:, :, idx]
    
            # QuadMesh stores an array of length (nx*ny) in flattened form.
            # Use ravel(order='C') to match the internal layout.
            pcm.set_array(data.ravel())
    
            # update color limits if autoscaling requested
            if vmin is None or vmax is None:
                new_vmin = float(np.nanmin(data)) if vmin is None else vmin
                new_vmax = float(np.nanmax(data)) if vmax is None else vmax
                # Avoid degenerate clim
                if np.isclose(new_vmin, new_vmax):
                    new_vmax = new_vmin + 1e-12
                pcm.set_clim(new_vmin, new_vmax)
    
            # tell the colorbar to rescale
            cb.update_normal(pcm)
    
            title.set_text(f"time = {self.solution_t[idx]:.4g}")
            return (pcm,)
    
        with writer.saving(fig, outpath, dpi=dpi):
            for i in range(n_frames):
                update_frame(i)
                # draw and grab
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                writer.grab_frame()
                if i % 20 == 0 or i == n_frames - 1:
                    print(f"Rendering animation frame {i+1}/{n_frames}")
    
        print(f"Saved animation to {outpath}")
        return outpath


        
    
    def total_mass_vs_time(self):
        """
        Plot total mass ∫∫ u(x,theta,t) dtheta dx versus time.
        """
        dx = self.L_x / self.mesh_points_x
        dtheta = self.L_theta / self.n_theta
    
        # total mass at each time
        mass = np.sum(self.u, axis=(0, 1)) * dx * dtheta
    
        fig, ax = plt.subplots()
        ax.plot(self.solution_t, mass, lw=2)
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\int u(x,\theta,t)\,d\theta\,dx$")
        ax.set_title(f"{self.file_tag}  total mass")
        plt.tight_layout()
    
        if self.save:
            outpath = os.path.join(self.folder_tag, f"{self.file_tag}_total_mass_vs_time.png")
            fig.savefig(outpath, dpi=200)
            print(f"Saved total-mass plot to {outpath}")
    
        return fig, ax

        
    def evolution_all_points(self, plot_every=1):
        """
        Plot time series for every (theta_j, x_i) point on the same axes.
        plot_every: only plot every `plot_every`-th theta and every `plot_every`-th x to reduce clutter.
        """
        fig, ax = plt.subplots()
        FigureCanvasAgg(fig)
        step = max(1, int(plot_every))
        n_plotted = 0
        for j in range(0, self.n_theta, step):
            for i in range(0, self.mesh_points_x, step):
                ax.plot(self.solution_t, self.u[j, i, :], lw=0.6, alpha=0.7)
                n_plotted += 1

        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$u(\theta,x,t)$ for every grid point")
        ax.set_title(f"{self.file_tag}  all points (plotted={n_plotted})")
        ax.axhline(y=0.0, color='black', linestyle='dashed')
        plt.tight_layout()

        if self.save:
            outpath = os.path.join(self.folder_tag, f"{self.file_tag}_evolution_all_points.png")
            fig.savefig(outpath, dpi=200)
            print(f"Saved evolution-all-points plot to {outpath}")
        return fig, ax

    def time_derivative_all_points(self, plot_every=1):
        """
        Finite-difference estimate of time derivative for every (theta_j, x_i) point.
        Plots d/dt u(...) for each point on the same axes.
        plot_every: subsample theta and x as above.
        """
        t = self.solution_t
        if t.size < 2:
            raise ValueError("Need at least two time points to compute derivative.")
        dt = np.diff(t)  # shape (n_time-1,)

        fig, ax = plt.subplots()
        FigureCanvasAgg(fig)
        step = max(1, int(plot_every))
        n_plotted = 0
        for j in range(0, self.n_theta, step):
            for i in range(0, self.mesh_points_x, step):
                deriv = (self.u[j, i, 1:] - self.u[j, i, :-1]) / dt
                ax.plot(t[1:], deriv, lw=0.6, alpha=0.7)
                n_plotted += 1

        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\partial_t u(\theta,x,t)$ for every grid point")
        ax.set_title(f"{self.file_tag}  time derivatives all points (plotted={n_plotted})")
        ax.axhline(y=0.0, color='black', linestyle='dashed')
        plt.tight_layout()

        if self.save:
            outpath = os.path.join(self.folder_tag, f"{self.file_tag}_time_derivative_all_points.png")
            fig.savefig(outpath, dpi=200)
            print(f"Saved time-derivative-all-points plot to {outpath}")
        return fig, ax
    
    
        
    
    def total_population_vs_x(self, time_index=-1, figsize=(8, 6), lw=4.0):
        """
        Plot total population density along x by summing over phenotype space.
    
        This computes:
            U(x,t) = ∫ u(x, theta, t) dtheta
        and plots U against x for a fixed time.
    
        Parameters
        ----------
        time_index : int
            Index into solution_t.
        figsize : tuple
            Figure size.
        lw : float
            Line width.
        """
        if time_index < 0:
            time_index = self.n_time + time_index
        if not (0 <= time_index < self.n_time):
            raise IndexError("time_index out of range")
    
        dtheta = self.L_theta / self.n_theta
        total_population = np.sum(self.u[:, :, time_index], axis=0) * dtheta
    
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.x_points, total_population, lw=lw, c="chocolate")
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\int_{V} u(t, x,\theta)\,d\theta$")
        ax.set_ylabel(r"total population") #debug
        
        #ax.set_title(rf"$t={self.solution_t[time_index]:.4g}$")
        ax.set_ylim(0., 3.)
        ax.set_xlim(0., self.L_x)
        ax.minorticks_off()
        ax.set_xticks([0, self.L_x/2, self.L_x])
        ax.set_yticks([0, 0.5, 1.])
        
        ax.set_box_aspect(0.9295303054)
        
        plt.tight_layout()
    
        if self.save:
            outpath = os.path.join(
                self.folder_tag,
                f"{self.file_tag}_total_population_vs_x_t{time_index}.pdf"
            )
            fig.savefig(outpath, dpi=200)
            print(f"Saved total-population-vs-x plot to {outpath}")
    
        return fig, ax
    
    def all_species_vs_x(self, time_index=-1, figsize=(8, 6), lw=4.0):
        """
        Plot density of each species, u_i, along x on the same graph for every i

        Parameters
        ----------
        time_index : int
            Index into solution_t.
        figsize : tuple
            Figure size.
        lw : float
            Line width.
        """
        if time_index < 0:
            time_index = self.n_time + time_index
        if not (0 <= time_index < self.n_time):
            raise IndexError("time_index out of range")
    
        fig, ax = plt.subplots(figsize=figsize)
        
        colours = ["firebrick", "forestgreen", "cornflowerblue"]
        
        i=0
        for u_i in self.u:
            if i in [0,1,2]:
                c=colours[i]
            else:
                c=None
            
            ax.plot(self.x_points, u_i[:,time_index], lw=lw, label=r"$u_{{%d}}(t,x)$" %(i+1), c=c)
            i+=1
    
        
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"Each $u_i(t,x)$")
        ax.set_ylabel(r"$\theta$") #debug
        ax.legend(loc="upper left")
        
        #ax.set_title(rf"$t={self.solution_t[time_index]:.4g}$")
        ax.set_ylim(0., 3.)
        ax.set_xlim(0., self.L_x)
        ax.minorticks_off()
        ax.set_xticks([0, self.L_x/2, self.L_x])
        ax.set_yticks([0, 1., 2., 3.])
        
        ax.set_box_aspect(0.9295303054)
        
        plt.tight_layout()
    
        if self.save:
            outpath = os.path.join(
                self.folder_tag,
                f"{self.file_tag}_all_species_vs_x_t{time_index}.pdf"
            )
            fig.savefig(outpath, dpi=200)
            print(f"Saved all_species_vs_x plot to {outpath}")
    
        return fig, ax
    
    
    def animate_total_population_vs_x(self, fps=10, dpi=150, filename="total_population_vs_x.mp4",
                                      sample_rate=1, quick_start=True, normalize=False,
                                      ylim=(0,1)):
        """
        Create and save an mp4 animation of the phenotype-integrated population U(x,t).
    
        Parameters
        ----------
        fps : int
            Frames per second.
        dpi : int
            Figure DPI for saved video.
        filename : str
            Output mp4 filename.
        sample_rate : int
            Take every `sample_rate`-th time point.
        quick_start : bool
            If True, skip the earliest frames similarly to `animate`.
        normalize : bool
            If True, divide each profile by its spatial integral so the shape is emphasized.
        ylim : tuple or None
            Optional y-axis limits.
        """

        if (quick_start and self.solution_t.size > 40*self.n_theta+1) :
            frame_indices_1 = list(range(0, 40 * self.n_theta, 20))
            frame_indices_2 = list(range(40 * self.n_theta, self.n_time, sample_rate))
            frame_indices = frame_indices_1 + frame_indices_2
        elif quick_start :
            frame_indices = list(range(0, self.n_time, 10*sample_rate))
        else:
            #frame_indices = list(range(0, self.n_time, sample_rate)) #debug
            frame_indices = list(range(0, 1001, sample_rate))
    
        n_frames = len(frame_indices)
        if n_frames == 0:
            raise ValueError("No frames to render (check sample_rate and solution_t)")
    
        dtheta = self.L_theta / self.n_theta
        profiles = np.sum(self.u, axis=0) * dtheta  # shape (mesh_points_x, n_time)
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])
        FigureCanvasAgg(fig)
        (line,) = ax.plot(self.x_points, profiles[:, frame_indices[0]], lw=4, c="chocolate")
        title = ax.set_title(f"time = {self.solution_t[frame_indices[0]]:.4g}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\int u(t, x, \theta)\,d\theta$")
        ax.set_xlim((0,self.L_x))
        
        ax.set_xticks([0, self.L_x/2, self.L_x])
        if ylim is not None:
            ax.set_ylim(ylim)
    
        def get_profile(idx):
            profile = profiles[:, idx]
            if normalize:
                area = np.trapz(profile, self.x_points)
                if not np.isclose(area, 0.0):
                    profile = profile / area
            return profile
    
        line.set_ydata(get_profile(frame_indices[0]))
    
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='PhenotypePlotter'), bitrate=4000)
        outpath = os.path.join(self.folder_tag, f"{self.file_tag}_{filename}")
    
        def update_frame(i):
            idx = frame_indices[i]
            line.set_ydata(get_profile(idx))
            title.set_text(f"time = {self.solution_t[idx]:.4g}")
            return (line,)
    
        with writer.saving(fig, outpath, dpi=dpi):
            for i in range(n_frames):
                update_frame(i)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                writer.grab_frame()
                if i % 20 == 0 or i == n_frames - 1:
                    print(f"Rendering animation frame {i+1}/{n_frames}")
    
        print(f"Saved animation to {outpath}")
        return outpath


    def kymograph_total_population(self, vmin=None, vmax=None, figsize=(8, 6)):
        """
        Kymograph of the phenotype-integrated population
    
            U(x,t) = ∫ u(x,θ,t) dθ
    
        with x on the horizontal axis and t on the vertical axis.
        """
        dtheta = self.L_theta / self.n_theta
        U = np.sum(self.u, axis=0) * dtheta   # shape (mesh_points_x, n_time)
    
        fig, ax = self._prepare_fig_axes(figsize=figsize)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", "5%", "5%")
        
        pcm = ax.pcolormesh(
            self.solution_t,
            self.x_points,
            U,                   
            shading="auto",
            cmap="viridis",
            rasterized=self.rasterize_heatmap,
            vmin=vmin,
            vmax=vmax,
        )
    
        cb = fig.colorbar(pcm, cax=cax)
    #    cb.set_label(r"$\int_{V} u(t, x,\theta)\,d\theta$")
        cb.set_label(r"$\overline{u}(t, x)$")
    
    
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        
        ax.set_yticks([0, self.L_x / 2, self.L_x])
        ax.minorticks_off()
    
        plt.tight_layout()
    
        if self.save:
            outpath = os.path.join(
                self.folder_tag,
                f"{self.file_tag}_kymograph_total_population.pdf",
            )
            fig.savefig(outpath, dpi=200)
            print(f"Saved kymograph to {outpath}")
    
        return fig, ax

    def animate_species_population_vs_x(
        self,
        fps=10,
        dpi=150,
        filename="species_population_vs_x.mp4",
        sample_rate=1,
        quick_start=True,
        ylim=(0, 2),
    ):
        """
        Create and save an mp4 animation showing each phenotype/species
        population density U_j(x,t) against x on the same axes.
        """
    
        if quick_start and self.solution_t.size > 40 * self.n_theta + 1:
            frame_indices_1 = list(range(0, 40 * self.n_theta, 20))
            frame_indices_2 = list(
                range(40 * self.n_theta, self.n_time, sample_rate)
            )
            frame_indices = frame_indices_1 + frame_indices_2
        elif quick_start:
            frame_indices = list(range(0, self.n_time, 10 * sample_rate))
        else:
            frame_indices = list(range(0, self.n_time, sample_rate))
            #frame_indices = list(range(0, 1001, sample_rate))
    
        n_frames = len(frame_indices)
        if n_frames == 0:
            raise ValueError("No frames to render (check sample_rate and solution_t)")
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_axes([AX_LEFT, AX_BOTTOM, AX_WIDTH, AX_HEIGHT])
        FigureCanvasAgg(fig)
    
        # Create one line for each species/phenotype
        lines = []
        colours = ["firebrick", "forestgreen", "cornflowerblue"]
        for j in range(self.n_theta):
            if j in [0,1,2]:
                colour = colours[j]
            else:
                colour = None
            (line,) = ax.plot(
                self.x_points,
                self.u[j, :, frame_indices[0]],
                lw=3,
                c=colour,
            )
            lines.append(line)
    
        title = ax.set_title(
            f"time = {self.solution_t[frame_indices[0]]:.4g}"
        )
    
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(t,x,\theta)$")
        ax.set_xlim(0, self.L_x)
    
        if ylim is not None:
            ax.set_ylim(ylim)
    
        Writer = animation.writers["ffmpeg"]
        writer = Writer(
            fps=fps,
            metadata=dict(artist="PhenotypePlotter"),
            bitrate=4000,
        )
    
        outpath = os.path.join(
            self.folder_tag,
            f"{self.file_tag}_{filename}",
        )
    
        def update_frame(i):
            idx = frame_indices[i]
    
            for j, line in enumerate(lines):
                line.set_ydata(self.u[j, :, idx])
    
            title.set_text(
                f"time = {self.solution_t[idx]:.4g}"
            )
    
            return tuple(lines) + (title,)
    
        with writer.saving(fig, outpath, dpi=dpi):
            for i in range(n_frames):
                update_frame(i)
                writer.grab_frame()
    
                if i % 20 == 0 or i == n_frames - 1:
                    print(
                        f"Rendering animation frame {i + 1}/{n_frames}"
                    )
    
        print(f"Saved animation to {outpath}")
        return outpath
    








if __name__ == "__main__":

    folder_tag = "results_paper/"
    folder_tag += "example_1"
    
    
    y_file_name = folder_tag + "/data_u_theta.csv"
    t_file_name = folder_tag + "/data_t.csv"
    parameter_file_name = folder_tag + "/parameters.txt"
    
    file_tag = ""
    save = True
    
    #Reading in parameters
    print("Reading in parameters...")
    with open(parameter_file_name) as param_file:
        parameters_string = param_file.read()

    #All the values we need are between '=' and '\n'
    #change all '\n' to '='
    parameters_string = parameters_string.replace('\n','=')
    #remove all white space
    parameters_string = ''.join(parameters_string.split())
    #take everything in between the '='
    parameters_list = parameters_string.split(sep='=')
    #only take every other element so we only take parameter values not names
    parameters_list = parameters_list[1::2]
    
    L_x = float(parameters_list[9])
    mesh_points_x = int(parameters_list[10])
    n_theta = int(parameters_list[0])
    L_theta = 1.  # always 1
    
    print("Loading solution data...")
    solution_y = np.loadtxt(y_file_name, delimiter=",")
    solution_t = np.loadtxt(t_file_name, delimiter=",")

    # ------------------------------------------------------------------
    # Create plotter
    # ------------------------------------------------------------------
    plotter = PhenotypePlotter(
        solution_y=solution_y,
        solution_t=solution_t,
        L_x=L_x,
        mesh_points_x=mesh_points_x,
        n_theta=n_theta,
        L_theta=L_theta,
        folder_tag=folder_tag,
        file_tag=file_tag,
        save=save,
        num_heatmap_ticks=3,
    )

    # ------------------------------------------------------------------
    # Produce plots
    # ------------------------------------------------------------------
    print("Creating time evolution graphs")
    plotter.evolution_all_points()
    plotter.time_derivative_all_points()
    
    print("Creating total mass evolution graph")
    plotter.total_mass_vs_time()
    
    print("Creating final-time heatmap...")
    plotter.final_snapshot(vmin=0, vmax=1.35)
    
    print("Creating kymograph...")
    plotter.kymograph_total_population(vmin=0, vmax=1., figsize=(32, 6))
    
    
    print("Creating total population vs x plot")
    plotter.total_population_vs_x(-1)
    

    print("Creating animation...")
    plotter.animate(
        fps=10,
        sample_rate=1,
        quick_start=True,
        filename="phenotype_evolution.mp4",
    )
    
    plt.close('all')

    print("Done.")
    
    plotter.animate_species_population_vs_x(
        fps=10,
        sample_rate=1,
        quick_start=False,
        filename="all_species_vs_x.mp4",
        ylim=[0.0,3.]
    )

    
    plt.close('all')

