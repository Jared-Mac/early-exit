import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from simulation import Simulation
import itertools
import pandas as pd
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from matplotlib.colors import LinearSegmentedColormap

def run_single_simulation(thresholds, cached_data_file, evaluation_time):
    """
    Helper function to run a single simulation with given parameters
    """
    t1, t2, t3 = thresholds
    last_threshold = min(thresholds)
    full_thresholds = [t1, t2, t3, last_threshold]
    
    sim = Simulation(strategy='early_exit_custom',
                    cached_data_file=cached_data_file,
                    confidence_thresholds=full_thresholds)
    
    result = sim.evaluate(max_sim_time=evaluation_time)
    
    return {
        'threshold1': t1,
        'threshold2': t2,
        'threshold3': t3,
        'accuracy': result['accuracy'],
        'avg_latency': result['avg_latency'],
        'exit_distribution': result['exit_numbers']
    }

def test_threshold_combinations(thresholds_range=np.arange(0.5, 1.0, 0.05),
                              cached_data_file='models/cifar10/resnet18/blocks/cached_logits.pkl',
                              evaluation_time=1000,
                              max_workers=4):
    """
    Test different combinations of confidence thresholds in parallel
    """
    combinations = list(itertools.product(thresholds_range, repeat=3))
    
    results = []
    # Process in smaller batches to manage memory
    batch_size = 10
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        
        run_sim_partial = partial(run_single_simulation, 
                                cached_data_file=cached_data_file,
                                evaluation_time=evaluation_time)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(run_sim_partial, batch))
            results.extend(batch_results)
            
    return pd.DataFrame(results)

def plot_metric_heatmaps(results_df, metric='accuracy'):
    """
    Create heatmaps showing specified metric for different threshold combinations
    """
    fig = plt.figure(figsize=(15, 5))
    
    threshold_pairs = [('threshold1', 'threshold2'),
                      ('threshold2', 'threshold3'),
                      ('threshold1', 'threshold3')]
    
    for idx, (x_thresh, y_thresh) in enumerate(threshold_pairs, 1):
        plt.subplot(1, 3, idx)
        
        pivot_data = results_df.pivot_table(
            values=metric,
            index=x_thresh,
            columns=y_thresh,
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'{metric.capitalize()}: {x_thresh} vs {y_thresh}')
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_heatmaps_cifar_10_mobilenetv2.png')
    plt.close()

def plot_3d_surface(results_df, metric='accuracy', thresh3_value=None):
    """
    Create 3D surface plot showing specified metric vs thresholds
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reduce resolution for visualization
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    
    # Filter for specific threshold3 value
    filtered_df = results_df[results_df['threshold3'] == thresh3_value]
    
    X, Y = np.meshgrid(thresh1, thresh2)
    values = np.zeros_like(X)
    
    for i, t1 in enumerate(thresh1):
        for j, t2 in enumerate(thresh2):
            mask = (filtered_df['threshold1'] == t1) & \
                   (filtered_df['threshold2'] == t2)
            if mask.any():
                values[j, i] = filtered_df[mask][metric].values[0]
    
    surf = ax.plot_surface(X, Y, values,
                          cmap='viridis',
                          antialiased=True)
    
    ax.set_xlabel('Threshold 1')
    ax.set_ylabel('Threshold 2')
    ax.set_zlabel(metric.capitalize())
    plt.colorbar(surf)
    
    plt.title(f'{metric.capitalize()} (Threshold 3 = {thresh3_value:.2f})')
    plt.savefig(f'threshold_{metric}_3d_surface_t3_{thresh3_value:.2f}.png')
    plt.close()
    # Explicitly clear memory
    plt.clf()
    plt.close('all')

def plot_3d_thresholds(results_df, metric='accuracy'):
    """
    Create an enhanced 3D scatter plot with contour projections for better visualization
    of the metric's topography across threshold combinations
    """
    # Create figure with larger size and specific projection
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create main scatter plot with enhanced visibility
    scatter = ax.scatter(results_df['threshold1'],
                        results_df['threshold2'],
                        results_df['threshold3'],
                        c=results_df[metric],
                        cmap='viridis',
                        s=100,
                        alpha=0.6)
    
    # Add contour projections on each face
    # XY plane (bottom)
    x = results_df['threshold1']
    y = results_df['threshold2']
    z = results_df['threshold3']
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = np.linspace(z.min(), z.max(), 100)
    
    Xi, Yi = np.meshgrid(xi, yi)
    Z_bottom = griddata((x, y), results_df[metric], (Xi, Yi), method='cubic')
    ax.contour(Xi, Yi, Z_bottom, 
               zdir='z', 
               offset=z.min(), 
               cmap='viridis',
               alpha=0.5)
    
    # XZ plane (back)
    Xi, Zi = np.meshgrid(xi, zi)
    Y_back = griddata((x, z), results_df[metric], (Xi, Zi), method='cubic')
    ax.contour(Xi, Y_back,
               Zi, 
               zdir='y', 
               offset=y.max(),
               cmap='viridis',
               alpha=0.5)
    
    # YZ plane (side)
    Yi, Zi = np.meshgrid(yi, zi)
    X_side = griddata((y, z), results_df[metric], (Yi, Zi), method='cubic')
    ax.contour(X_side,
               Yi, 
               Zi,
               zdir='x', 
               offset=x.min(),
               cmap='viridis',
               alpha=0.5)
    
    # Enhance the plot with better labels and styling
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(f'{metric.capitalize()}', rotation=270, labelpad=15, size=12)
    
    ax.set_xlabel('Threshold 1', labelpad=10, size=12)
    ax.set_ylabel('Threshold 2', labelpad=10, size=12)
    ax.set_zlabel('Threshold 3', labelpad=10, size=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add gridlines for better depth perception
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f'Threshold Combinations Impact on {metric.capitalize()}\nwith Contour Projections', 
              pad=20, 
              size=14)
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_3d_scatter_enhanced.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

# Alternative version using size to represent metric as well
def plot_3d_thresholds_with_size(results_df, metric='accuracy'):
    """
    Create a 3D scatter plot where both color and point size
    represent the metric value for additional emphasis
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize metric values for size scaling
    size_scale = (results_df[metric] - results_df[metric].min()) / \
                 (results_df[metric].max() - results_df[metric].min())
    sizes = 50 + size_scale * 200  # Scale sizes between 50 and 250
    
    scatter = ax.scatter(results_df['threshold1'],
                        results_df['threshold2'],
                        results_df['threshold3'],
                        c=results_df[metric],
                        s=sizes,
                        cmap='viridis',
                        alpha=0.6)
    
    colorbar = plt.colorbar(scatter)
    colorbar.set_label(f'{metric.capitalize()}', rotation=270, labelpad=15)
    
    ax.set_xlabel('Threshold 1')
    ax.set_ylabel('Threshold 2')
    ax.set_zlabel('Threshold 3')
    plt.title(f'Threshold Combinations Colored by {metric.capitalize()}\n(Point size also indicates {metric})')
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_3d_scatter_sized.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def plot_3d_topology(results_df, metric='accuracy'):
    """
    Create a 3D surface visualization showing the topography of the metric across
    threshold combinations using triangulation and isolines
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates and values
    points = results_df[['threshold1', 'threshold2', 'threshold3']].values
    values = results_df[metric].values
    
    # Create triangulation
    tri = Delaunay(points)
    
    # Create the triangular mesh
    mesh = ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2],
                          triangles=tri.simplices,
                          cmap='viridis',
                          alpha=0.8,
                          linewidth=0.2,
                          edgecolor='white',
                          shade=True)
    
    # Add isolines on the surface
    levels = np.linspace(values.min(), values.max(), 10)
    for level in levels:
        mask = np.abs(values - level) < (values.max() - values.min()) * 0.05
        if np.any(mask):
            ax.plot_trisurf(points[mask, 0], 
                           points[mask, 1], 
                           points[mask, 2],
                           color='black',
                           alpha=0.1,
                           linewidth=0.5)
    
    # Add contour projections on walls
    ax.contour(points[:, 0], points[:, 1], values,
               zdir='z',
               offset=points[:, 2].min(),
               cmap='viridis',
               alpha=0.5,
               levels=levels)
    
    # Enhance visual appearance
    colorbar = plt.colorbar(mesh)
    colorbar.set_label(f'{metric.capitalize()}', rotation=270, labelpad=15, size=12)
    
    ax.set_xlabel('Threshold 1', labelpad=10, size=12)
    ax.set_ylabel('Threshold 2', labelpad=10, size=12)
    ax.set_zlabel('Threshold 3', labelpad=10, size=12)
    
    # Set optimal viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.title(f'Topographical Surface of {metric.capitalize()}\nacross Threshold Combinations', 
              pad=20, 
              size=14)
    
    # Add text annotations for local maxima/minima
    extreme_points = pd.concat([
        results_df.nlargest(3, metric),
        results_df.nsmallest(3, metric)
    ])
    
    for _, point in extreme_points.iterrows():
        ax.text(point['threshold1'], 
                point['threshold2'], 
                point['threshold3'],
                f'{point[metric]:.3f}',
                size=8)
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_topology.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def plot_stacked_3d_surfaces(results_df, metric='accuracy'):
    """
    Create a single 3D plot with stacked surfaces for different threshold3 values
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique threshold3 values
    thresh3_values = sorted(results_df['threshold3'].unique())
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    
    # Create color map with different alpha values for each surface
    n_surfaces = len(thresh3_values)
    colors = plt.cm.viridis(np.linspace(0, 1, n_surfaces))
    
    X, Y = np.meshgrid(thresh1, thresh2)
    surfaces = []
    
    # Plot each surface
    for idx, thresh3 in enumerate(thresh3_values):
        filtered_df = results_df[results_df['threshold3'] == thresh3]
        values = np.zeros_like(X)
        
        for i, t1 in enumerate(thresh1):
            for j, t2 in enumerate(thresh2):
                mask = (filtered_df['threshold1'] == t1) & \
                       (filtered_df['threshold2'] == t2)
                if mask.any():
                    values[j, i] = filtered_df[mask][metric].values[0]
        
        # Plot surface with custom color and transparency
        surf = ax.plot_surface(X, Y, values,
                             color=colors[idx],
                             alpha=0.3,
                             label=f'T3={thresh3:.2f}')
        surfaces.append(surf)
    
    # Customize the plot
    ax.set_xlabel('Threshold 1')
    ax.set_ylabel('Threshold 2')
    ax.set_zlabel(metric.capitalize())
    
    # Add legend
    legend_surfaces = [plt.Rectangle((0,0),1,1, fc=colors[i], alpha=0.3) 
                      for i in range(n_surfaces)]
    ax.legend(legend_surfaces, 
             [f'T3={t:.2f}' for t in thresh3_values],
             loc='upper right')
    
    plt.title(f'Stacked {metric.capitalize()} Surfaces for Different Threshold 3 Values')
    
    # Set optimal viewing angle
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_stacked_surfaces.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def plot_dynamic_3d_surfaces(results_df, metric='accuracy'):
    """
    Create an enhanced 3D visualization with contours and topographical emphasis
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    thresh3_values = sorted(results_df['threshold3'].unique())
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    
    X, Y = np.meshgrid(thresh1, thresh2)
    
    # Create offset multiplier for vertical spacing
    z_offset_multiplier = 0.1 * (results_df[metric].max() - results_df[metric].min())
    
    # Custom colormap for topographical emphasis
    colors = plt.cm.terrain(np.linspace(0, 1, 256))
    custom_cmap = LinearSegmentedColormap.from_list('custom_terrain', colors)
    
    surfaces = []
    for idx, thresh3 in enumerate(thresh3_values):
        filtered_df = results_df[results_df['threshold3'] == thresh3]
        Z = np.zeros_like(X)
        
        for i, t1 in enumerate(thresh1):
            for j, t2 in enumerate(thresh2):
                mask = (filtered_df['threshold1'] == t1) & \
                       (filtered_df['threshold2'] == t2)
                if mask.any():
                    Z[j, i] = filtered_df[mask][metric].values[0]
        
        # Add vertical offset for each surface
        Z_offset = Z + idx * z_offset_multiplier
        
        # Plot main surface with custom shading
        surf = ax.plot_surface(X, Y, Z_offset,
                             cmap=custom_cmap,
                             linewidth=0.5,
                             antialiased=True,
                             alpha=0.8)
        
        # Create meshgrids for contour projections
        X_wall = np.tile(thresh1, (len(thresh2), 1))
        Y_wall = np.tile(thresh2, (len(thresh1), 1)).T
        
        # Add contour projections on the walls
        # XZ contour (side wall)
        ax.contour(X_wall, 
                  np.ones_like(X_wall) * (Y.min() - (Y.max() - Y.min()) * 0.2),
                  Z_offset,
                  zdir='y',
                  offset=Y.min() - (Y.max() - Y.min()) * 0.2,
                  cmap='viridis',
                  alpha=0.5)
        
        # YZ contour (back wall)
        ax.contour(np.ones_like(Y_wall) * (X.min() - (X.max() - X.min()) * 0.2),
                  Y_wall,
                  Z_offset,
                  zdir='x',
                  offset=X.min() - (X.max() - X.min()) * 0.2,
                  cmap='viridis',
                  alpha=0.5)
        
        surfaces.append(surf)
    
    # Enhance the appearance
    ax.set_xlabel('Threshold 1', labelpad=20)
    ax.set_ylabel('Threshold 2', labelpad=20)
    ax.set_zlabel(metric.capitalize(), labelpad=20)
    
    # Add colorbar
    fig.colorbar(surfaces[-1], ax=ax, shrink=0.5, aspect=5,
                label=f'{metric.capitalize()} Value')
    
    plt.title(f'Topographical {metric.capitalize()} Surfaces\nfor Different Threshold 3 Values',
              pad=20, size=12)
    
    # Set dynamic viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Adjust axis limits to accommodate projections
    ax.set_xlim(X.min() - (X.max() - X.min()) * 0.2, X.max())
    ax.set_ylim(Y.min() - (Y.max() - Y.min()) * 0.2, Y.max())
    
    plt.tight_layout()
    plt.savefig(f'threshold_{metric}_topographical.png',
                dpi=300,
                bbox_inches='tight')
    plt.close()

def plot_stacked_3d_surfaces_side_by_side(results_df):
    """
    Create publication-quality 2x3 grid of 3D plots with larger dimensions
    """
    # Set up publication-quality plotting parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,    # Increased from 10
        'font.size': 12,         # Increased from 10
        'legend.fontsize': 10,   # Increased from 8
        'xtick.labelsize': 10,   # Increased from 8
        'ytick.labelsize': 10    # Increased from 8
    })
    
    # Select evenly spaced threshold3 values
    all_thresh3 = sorted(results_df['threshold3'].unique())
    target_values = [0.65, 0.75, 0.85]
    
    selected_thresh3 = []
    for target in target_values:
        closest_val = min(all_thresh3, key=lambda x: abs(x - target))
        selected_thresh3.append(closest_val)
    
    # Create figure with larger size and adjusted grid spacing
    fig = plt.figure(figsize=(15, 10))
    
    # Adjust grid spacing to move colorbar right
    gs = fig.add_gridspec(2, 4, 
                         width_ratios=[1, 1, 1, 0.15],  # Last value controls colorbar width
                         wspace=0.4)  # Increased from default to add space before colorbar
    
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    X, Y = np.meshgrid(thresh1, thresh2)
    
    metrics = ['accuracy', 'avg_latency']
    metric_ranges = {metric: {'min': results_df[metric].min(),
                            'max': results_df[metric].max()} 
                    for metric in metrics}
    
    for metric_idx, metric in enumerate(metrics):
        cmap = 'viridis' if metric == 'accuracy' else 'RdYlGn_r'
        
        # Create colorbar
        norm = plt.Normalize(metric_ranges[metric]['min'], 
                           metric_ranges[metric]['max'])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar_ax = fig.add_subplot(gs[metric_idx, -1])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        metric_name = 'Accuracy' if metric == 'accuracy' else 'Latency'
        cbar.set_label(f'{metric_name}', size=12)
        
        for idx, thresh3 in enumerate(selected_thresh3):
            ax = fig.add_subplot(gs[metric_idx, idx], projection='3d')
            
            filtered_df = results_df[results_df['threshold3'] == thresh3]
            values = np.zeros_like(X)
            
            for i, t1 in enumerate(thresh1):
                for j, t2 in enumerate(thresh2):
                    mask = (filtered_df['threshold1'] == t1) & \
                           (filtered_df['threshold2'] == t2)
                    if mask.any():
                        values[j, i] = filtered_df[mask][metric].values[0]
            
            surf = ax.plot_surface(X, Y, values,
                                 cmap=cmap,
                                 alpha=0.8,
                                 vmin=metric_ranges[metric]['min'],
                                 vmax=metric_ranges[metric]['max'])
            
            ax.set_xlabel('Threshold 1', labelpad=10)
            ax.set_ylabel('Threshold 2', labelpad=10)
            # ax.set_zlabel('Value', labelpad=10)
            
            ax.set_title(f'Threshold 3 = {thresh3:.2f}', pad=10, size=12)
            ax.view_init(elev=20, azim=135)
            ax.set_box_aspect([1, 1, 0.7])
            
            # Reduce tick density
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.zaxis.set_major_locator(plt.MaxNLocator(5))
    
    # # Add row labels on the left side
    # fig.text(0.02, 0.75, 'Accuracy', size=12, rotation=90)
    # fig.text(0.02, 0.25, 'Latency', size=12, rotation=90)
    
    plt.suptitle('Early Exit Threshold Impact', 
                 y=0.95, 
                 size=14)
    
    plt.tight_layout(rect=[0, 0, 0.95, 0.95], h_pad=2, w_pad=2)  # Adjusted right margin
    
    plt.savefig('threshold_metrics_comparison.pdf', 
                dpi=300, 
                bbox_inches='tight',
                format='pdf')
    plt.savefig('threshold_metrics_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def main():
    # Get dataset name and model from cache path
    cache_path = 'models/visualwakewords/mobilenetv2/blocks/cached_logits.pkl'
    dataset = cache_path.split('/')[1]
    model = cache_path.split('/')[2]
    
    # Run simulation with grid search over thresholds
    sim = Simulation(strategy='early_exit_custom', 
                    cached_data_file=cache_path)

    # Create thresholds grid
    thresholds = np.arange(0.5, 1.0, 0.05)
    results = []

    # Grid search over thresholds
    for t1 in thresholds:
        for t2 in thresholds:
            for t3 in thresholds:
                metrics = sim.run(confidence_thresholds=[t1, t2, t3, 1.0])
                results.append({
                    'threshold1': t1,
                    'threshold2': t2, 
                    'threshold3': t3,
                    'accuracy': metrics['accuracy'],
                    'avg_latency': metrics['avg_latency'],
                    'exit_distribution': str(metrics['exit_distribution'])
                })

    # Save results
    results_df = pd.DataFrame(results)
    output_file = f'threshold_results_{dataset}_{model}.csv'
    results_df.to_csv(output_file, index=False)
    # results_df = pd.read_csv('threshold_results.csv')
    # results_df = results_df[
    #     (results_df['threshold1'] >= 0.65) & (results_df['threshold1'] <= 0.95) &
    #     (results_df['threshold2'] >= 0.65) & (results_df['threshold2'] <= 0.95) &
    #     (results_df['threshold3'] >= 0.65) & (results_df['threshold3'] <= 0.95)
    # ]
    plot_stacked_3d_surfaces_side_by_side(results_df)

if __name__ == "__main__":
    main()
