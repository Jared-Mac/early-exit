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
import argparse
import os

def run_single_simulation(thresholds, cached_data_file, evaluation_time):
    """
    Helper function to run a single simulation with given parameters
    """
    t1, t2, t3 = thresholds
    last_threshold = min(thresholds)
    full_thresholds = [t1, t2, t3, last_threshold]
    
    sim = Simulation(strategy='early_exit_custom',
                    cached_data_file=cached_data_file,
                    confidence_thresholds=full_thresholds,
                    num_classes=10)
    
    result = sim.evaluate(max_sim_time=evaluation_time)
    
    return {
        'threshold1': t1,
        'threshold2': t2,
        'threshold3': t3,
        'accuracy': result['accuracy'],
        'avg_latency': result['avg_latency'],
        'exit_distribution': result['exit_numbers']
    }

def test_threshold_combinations(thresholds_range=None,
                              cached_data_file='models/cifar10/resnet18/blocks/cached_logits.pkl',
                              evaluation_time=1000,
                              max_workers=4):
    """
    Test different combinations of confidence thresholds in parallel
    """
    # Use linspace instead of arange for better floating point precision
    if thresholds_range is None:
        thresholds_range = np.linspace(0.5, 1.0, num=11)  # This gives: 0.5, 0.55, 0.6, ..., 1.0
    
    # Generate all combinations without filtering
    combinations = list(itertools.product(thresholds_range, repeat=3))
    
    results = []
    # Process in smaller batches to manage memory
    batch_size = 15
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

def plot_stacked_3d_surfaces_side_by_side(results_df, dql_results_df=None):
    """
    Create publication-quality 2x3 grid of 3D plots with DQL logit confidence overlay
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Create figure with larger size and adjusted grid spacing
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, 
                         width_ratios=[1, 1, 1, 0.15],
                         wspace=0.4)
    
    # Selected threshold3 values
    all_thresh3 = sorted(results_df['threshold3'].unique())
    target_values = [0.65, 0.75, 0.85]
    selected_thresh3 = []
    for target in target_values:
        closest_val = min(all_thresh3, key=lambda x: abs(x - target))
        selected_thresh3.append(closest_val)
    
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    X, Y = np.meshgrid(thresh1, thresh2)
    
    metrics = ['accuracy', 'avg_latency']
    metric_ranges = {metric: {'min': results_df[metric].min(),
                            'max': results_df[metric].max()} 
                    for metric in metrics}
    
    # If DQL results provided, update ranges
    if dql_results_df is not None:
        for metric in metrics:
            metric_ranges[metric]['min'] = min(metric_ranges[metric]['min'], 
                                             dql_results_df[metric].min())
            metric_ranges[metric]['max'] = max(metric_ranges[metric]['max'], 
                                             dql_results_df[metric].max())
    
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
            
            # Plot threshold-based surface
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
                                 alpha=0.7,
                                 vmin=metric_ranges[metric]['min'],
                                 vmax=metric_ranges[metric]['max'])
            
            # Overlay DQL results if provided
            if dql_results_df is not None:
                dql_mean = dql_results_df[metric].mean()
                dql_std = dql_results_df[metric].std()
                
                # Get exit logits and their frequencies for each block
                if 'exit_logits' in dql_results_df.columns:
                    try:
                        # Convert string representation of list to actual list
                        exit_logits = dql_results_df['exit_logits'].tolist()
                        # Create scatter points for DQL decisions
                        scatter_x = []
                        scatter_y = []
                        scatter_z = []
                        confidence_values = []
                        exit_points = []
                        
                        # Process each exit decision
                        for logits_list in exit_logits:
                            if isinstance(logits_list, list):
                                for block_idx, conf in enumerate(logits_list):
                                    # if conf > 0:  # Only show actual exit points
                                    scatter_x.append(conf)  # Use confidence as x coordinate
                                    scatter_y.append(block_idx / 3)  # Normalize block index for y coordinate
                                    scatter_z.append(dql_mean)
                                    confidence_values.append(conf)
                                    exit_points.append(block_idx)
                    
                        # Plot scatter points colored by confidence value
                        if scatter_x:
                            scatter = ax.scatter(scatter_x, scatter_y, scatter_z,
                                              c=confidence_values,
                                              s=100,
                                              cmap='plasma',
                                              alpha=0.6,
                                              label='DQL Exit Confidences')
                            
                            # Add colorbar for confidence values
                            if idx == len(selected_thresh3) - 1:
                                cbar = plt.colorbar(scatter, ax=ax)
                                cbar.set_label('Exit Confidence')
                    except Exception as e:
                        print(f"Error processing exit logits: {e}")
                
                # Add text annotation for average performance
                ax.text(X.min(), Y.max(), dql_mean,
                       f'DQL Mean: {dql_mean:.3f}',
                       fontsize=8)
            
            ax.set_xlabel('Threshold 1', labelpad=10)
            ax.set_ylabel('Threshold 2', labelpad=10)
            
            ax.set_title(f'Threshold 3 = {thresh3:.2f}', pad=10, size=12)
            ax.view_init(elev=20, azim=135)
            ax.set_box_aspect([1, 1, 0.7])
            
            # Reduce tick density
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.zaxis.set_major_locator(plt.MaxNLocator(5))
            

    plt.suptitle('Threshold-based vs DQL Performance Comparison', 
                 y=0.95, 
                 size=14)
    
    plt.savefig('threshold_dql_comparison_3d.pdf', 
                dpi=300, 
                bbox_inches='tight',
                format='pdf')
    plt.savefig('threshold_dql_comparison_3d.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def parse_exit_dist(dist_str):
    """
    Parse exit distribution from string format
    Handles both list format and dict format
    """
    try:
        if isinstance(dist_str, str):
            if dist_str.startswith('{'):  # Dictionary format
                # Convert string to dictionary
                dist_dict = eval(dist_str)
                # Convert block0-3 format to list
                return [dist_dict.get(f'block{i}', 0) for i in range(4)]
            elif dist_str.startswith('['):  # List format
                # Remove brackets and split by comma
                dist_str = dist_str.strip('[]').replace(' ', '')
                return [int(x) for x in dist_str.split(',') if x]
            else:
                print(f"Unexpected exit distribution format: {dist_str}")
                return [0, 0, 0, 0]
        elif isinstance(dist_str, list):
            return dist_str
        else:
            print(f"Unexpected type for exit_distribution: {type(dist_str)}")
            return [0, 0, 0, 0]
    except Exception as e:
        print(f"Error parsing exit distribution: {e}")
        print(f"Problem value: {dist_str}")
        return [0, 0, 0, 0]

def plot_exit_distribution(results_df):
    """
    Create a visualization showing exit distribution across different threshold combinations
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    # Create figure with subplots for each exit point
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.3)
    
    # Calculate exit percentages for each threshold combination
    results_df['exit_numbers'] = results_df['exit_distribution'].apply(parse_exit_dist)
    total_samples = results_df['exit_numbers'].apply(lambda x: sum(x))
    
    # Calculate percentage for each exit
    for i in range(4):
        results_df[f'exit_{i}_pct'] = results_df['exit_numbers'].apply(
            lambda x: (x[i] / sum(x)) * 100 if sum(x) > 0 else 0
        )
    
    # Selected threshold3 values
    all_thresh3 = sorted(results_df['threshold3'].unique())
    target_values = [0.65, 0.75, 0.85]
    selected_thresh3 = []
    for target in target_values:
        closest_val = min(all_thresh3, key=lambda x: abs(x - target))
        selected_thresh3.append(closest_val)
    
    thresh1 = sorted(results_df['threshold1'].unique())
    thresh2 = sorted(results_df['threshold2'].unique())
    X, Y = np.meshgrid(thresh1, thresh2)
    
    # Plot exit distribution for each threshold3 value
    for exit_point in range(4):
        ax = fig.add_subplot(2, 2, exit_point + 1, projection='3d')
        
        for thresh3 in selected_thresh3:
            filtered_df = results_df[results_df['threshold3'] == thresh3]
            values = np.zeros_like(X)
            
            for i, t1 in enumerate(thresh1):
                for j, t2 in enumerate(thresh2):
                    mask = (filtered_df['threshold1'] == t1) & \
                           (filtered_df['threshold2'] == t2)
                    if mask.any():
                        values[j, i] = filtered_df[mask][f'exit_{exit_point}_pct'].values[0]
            
            surf = ax.plot_surface(X, Y, values,
                                 cmap='viridis',
                                 alpha=0.7,
                                 label=f'T3={thresh3:.2f}')
        
        ax.set_xlabel('Threshold 1', labelpad=10)
        ax.set_ylabel('Threshold 2', labelpad=10)
        ax.set_zlabel('Exit %', labelpad=10)
        ax.set_title(f'Exit Point {exit_point + 1}', pad=10, size=12)
        ax.view_init(elev=20, azim=135)
        ax.set_box_aspect([1, 1, 0.7])
    
    plt.suptitle('Exit Distribution Across Threshold Combinations', 
                 y=0.95, 
                 size=14)
    
    plt.savefig('exit_distribution.pdf', 
                dpi=300, 
                bbox_inches='tight',
                format='pdf')
    plt.savefig('exit_distribution.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def plot_dql_comparison(results_df, dql_results_df):
    """
    Create visualization comparing DQL agent exit distribution with threshold-based approach
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], wspace=0.3, hspace=0.3)
    
    # Process threshold-based results
    results_df['exit_numbers'] = results_df['exit_distribution'].apply(parse_exit_dist)
    for i in range(4):
        results_df[f'exit_{i}_pct'] = results_df['exit_numbers'].apply(
            lambda x: (x[i] / sum(x)) * 100 if sum(x) > 0 else 0
        )
    
    # Process DQL results
    dql_results_df['exit_numbers'] = dql_results_df['exit_distribution'].apply(parse_exit_dist)
    for i in range(4):
        dql_results_df[f'exit_{i}_pct'] = dql_results_df['exit_numbers'].apply(
            lambda x: (x[i] / sum(x)) * 100 if sum(x) > 0 else 0
        )
    
    # Plot 1: Exit Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate average exit distributions
    threshold_exits = [results_df[f'exit_{i}_pct'].mean() for i in range(4)]
    dql_exits = [dql_results_df[f'exit_{i}_pct'].mean() for i in range(4)]
    
    x = np.arange(4)
    width = 0.35
    
    ax1.bar(x - width/2, threshold_exits, width, label='Threshold-based', alpha=0.8)
    ax1.bar(x + width/2, dql_exits, width, label='DQL Agent', alpha=0.8)
    
    ax1.set_ylabel('Exit Percentage (%)')
    ax1.set_xlabel('Exit Point')
    ax1.set_title('Average Exit Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Exit 1', 'Exit 2', 'Exit 3', 'Exit 4'])
    ax1.legend()
    
    # Plot 2: Accuracy vs Latency Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.scatter(results_df['avg_latency'], results_df['accuracy'], 
                alpha=0.5, label='Threshold-based', s=50)
    ax2.scatter(dql_results_df['avg_latency'], dql_results_df['accuracy'],
                alpha=0.5, label='DQL Agent', s=50)
    
    ax2.set_xlabel('Average Latency')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Latency Trade-off')
    ax2.legend()
    
    # Plot 3: Exit Point Usage Over Time (DQL)
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Assuming we have timestamps in DQL results
    if 'timestamp' in dql_results_df.columns:
        for i in range(4):
            ax3.plot(dql_results_df['timestamp'], 
                    dql_results_df[f'exit_{i}_pct'].rolling(window=10).mean(),
                    label=f'Exit {i+1}')
    
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Exit Usage (%)')
        ax3.set_title('DQL Exit Point Usage Over Time')
        ax3.legend()
    
    # Plot 4: Performance Metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    metrics = ['accuracy', 'avg_latency']
    threshold_metrics = [results_df[m].mean() for m in metrics]
    dql_metrics = [dql_results_df[m].mean() for m in metrics]
    
    # Normalize metrics for comparison
    max_vals = np.maximum(threshold_metrics, dql_metrics)
    threshold_metrics_norm = threshold_metrics / max_vals
    dql_metrics_norm = dql_metrics / max_vals
    
    x = np.arange(len(metrics))
    ax4.bar(x - width/2, threshold_metrics_norm, width, label='Threshold-based', alpha=0.8)
    ax4.bar(x + width/2, dql_metrics_norm, width, label='DQL Agent', alpha=0.8)
    
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Performance Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Accuracy', 'Latency'], rotation=45)
    ax4.legend()
    
    plt.suptitle('DQL Agent vs Threshold-based Approach Comparison', 
                 y=0.95, 
                 size=14)
    
    plt.savefig('dql_comparison.pdf', 
                dpi=300, 
                bbox_inches='tight',
                format='pdf')
    plt.savefig('dql_comparison.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def plot_logit_confidence_impact(results_df, dql_results_df):
    """
    Create visualization showing DQL agent's confidence-based decisions
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'axes.labelsize': 12,
        'font.size': 12,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    # Get best threshold configuration
    best_thresholds = results_df.nlargest(1, 'accuracy')[
        ['threshold1', 'threshold2', 'threshold3']].iloc[0]
    
    # Process DQL results
    dql_results_df['exit_numbers'] = dql_results_df['exit_distribution'].apply(parse_exit_dist)
    for i in range(4):
        dql_results_df[f'exit_{i}_pct'] = dql_results_df['exit_numbers'].apply(
            lambda x: (x[i] / sum(x)) * 100 if sum(x) > 0 else 0)
    
    # Plot 1: Exit Distribution
    ax1 = fig.add_subplot(gs[0])
    exit_points = range(4)
    dql_exits = [dql_results_df[f'exit_{i}_pct'].mean() for i in exit_points]
    
    ax1.bar(exit_points, dql_exits, alpha=0.6, color='blue')
    for i, thresh in enumerate(best_thresholds):
        ax1.axhline(y=thresh*100, color='red', linestyle='--', alpha=0.5,
                   label=f'Threshold {i+1}: {thresh:.2f}')
    
    ax1.set_xlabel('Exit Point')
    ax1.set_ylabel('Exit Usage (%)')
    ax1.set_title('DQL Exit Distribution')
    ax1.set_xticks(exit_points)
    ax1.set_xticklabels([f'Exit {i+1}' for i in exit_points])
    ax1.legend()
    
    # Plot 2: Accuracy Comparison
    ax2 = fig.add_subplot(gs[1])
    
    # Box plot of threshold-based accuracies
    threshold_acc = results_df['accuracy']
    dql_acc = dql_results_df['accuracy']
    
    data = [threshold_acc, dql_acc]
    labels = ['Threshold', 'DQL']
    
    ax2.boxplot(data, labels=labels)
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Distribution')
    
    # Plot 3: Latency Comparison
    ax3 = fig.add_subplot(gs[2])
    
    # Box plot of latencies
    threshold_lat = results_df['avg_latency']
    dql_lat = dql_results_df['avg_latency']
    
    data = [threshold_lat, dql_lat]
    labels = ['Threshold', 'DQL']
    
    ax3.boxplot(data, labels=labels)
    ax3.set_ylabel('Latency')
    ax3.set_title('Latency Distribution')
    
    plt.suptitle('DQL Agent Decision Analysis', y=1.05, size=14)
    
    plt.savefig('dql_decision_analysis.pdf', 
                dpi=300, 
                bbox_inches='tight',
                format='pdf')
    plt.savefig('dql_decision_analysis.png', 
                dpi=300, 
                bbox_inches='tight')
    plt.close()
    plt.clf()
    plt.close('all')

def main():
    """
    Run simulation with grid search over thresholds for specified dataset and model
    """
    parser = argparse.ArgumentParser(description='Run threshold analysis simulation')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., cifar10, visualwakewords)')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., resnet18, mobilenetv2)')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('--threshold-step', type=float, default=0.05, help='Step size for threshold grid search')
    parser.add_argument('--threshold-min', type=float, default=0.5, help='Minimum threshold value')
    parser.add_argument('--threshold-max', type=float, default=1.0, help='Maximum threshold value')
    parser.add_argument('--eval-time', type=int, default=1000, help='Evaluation time for each simulation')
    parser.add_argument('--dql-model', type=str, help='Path to trained DQL model')
    
    args = parser.parse_args()
    
    # Construct cache path
    cache_path = f'models/{args.dataset}/{args.model}/blocks/cached_logits.pkl'
    results_path = f'models/{args.dataset}/{args.model}/threshold_results_{args.dataset}_{args.model}.csv'
    print(f"Results path: {results_path}")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at {cache_path}")
        
    if os.path.exists(results_path):
        print(f"Loading existing results from {results_path}")
        results_df = pd.read_csv(results_path)
    else:
        print(f"Running simulation for {args.dataset} dataset with {args.model} model")
        print(f"Using cache from: {cache_path}")
        
        # Create thresholds grid
        thresholds = np.arange(args.threshold_min, args.threshold_max, args.threshold_step)
        results = []
        total_combinations = len(list(itertools.product(thresholds, repeat=3)))
        
        print(f"Testing {total_combinations} threshold combinations...")
        
        # Run simulations with progress tracking
        for i, (t1, t2, t3) in enumerate(itertools.product(thresholds, repeat=3)):
            if i % 10 == 0:  # Print progress every 10 combinations
                print(f"Progress: {i}/{total_combinations} combinations ({i/total_combinations*100:.1f}%)")
            
            # Initialize simulation with thresholds in original order
            sim = Simulation(
                strategy='early_exit_custom',
                cached_data_file=cache_path,
                confidence_thresholds=[t1, t2, t3, min([t1, t2, t3])],
                num_classes=args.num_classes
            )
            
            # Run evaluation
            metrics = sim.evaluate(max_sim_time=args.eval_time)
            
            # Store results
            results.append({
                'threshold1': t1,
                'threshold2': t2,
                'threshold3': t3,
                'accuracy': metrics['accuracy'],
                'avg_latency': metrics['avg_latency'],
                'exit_distribution': str(metrics['exit_numbers'])
            })
        
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        # Save as pickle to preserve data types
        results_df.to_pickle(results_path)
    
    # Also save as CSV for human readability
    output_file = f'models/{args.dataset}/{args.model}/threshold_results_{args.dataset}_{args.model}.csv'
    results_df.to_csv(output_file, index=False)
    
    if args.dql_model:
        print("\nEvaluating DQL agent...")
        dql_results = []
        
        # Initialize DQL simulation
        sim = Simulation(
            strategy='dql',
            cached_data_file=cache_path,
            load_model=args.dql_model,
            num_classes=args.num_classes
        )
        
        # Run evaluation and collect exit logits
        metrics = sim.evaluate(max_sim_time=args.eval_time)
        
        # Format metrics including exit logits
        result_dict = {
            'accuracy': metrics['accuracy'],
            'avg_latency': metrics['avg_latency'],
            'exit_distribution': str(metrics['exit_numbers'])
        }
        
        # Add exit logits if available
        if 'exit_logits' in metrics and metrics['exit_logits']:
            result_dict['exit_logits'] = str(metrics['exit_logits'])
        
        dql_results.append(result_dict)
        dql_results_df = pd.DataFrame(dql_results)
        
        # Save DQL results
        dql_output_file = f'dql_results_{args.dataset}_{args.model}.csv'
        dql_results_df.to_csv(dql_output_file, index=False)
        print(f"DQL results saved to {dql_output_file}")
        
        # Print DQL performance statistics
        print("\nDQL Agent Performance:")
        print(f"Average accuracy: {dql_results_df['accuracy'].mean():.3f} ± {dql_results_df['accuracy'].std():.3f}")
        print(f"Average latency: {dql_results_df['avg_latency'].mean():.3f} ± {dql_results_df['avg_latency'].std():.3f}")
        
        # Calculate and print exit distribution statistics for DQL
        dql_results_df['exit_numbers'] = dql_results_df['exit_distribution'].apply(parse_exit_dist)
        for i in range(4):
            dql_results_df[f'exit_{i}_pct'] = dql_results_df['exit_numbers'].apply(
                lambda x: (x[i] / sum(x)) * 100 if sum(x) > 0 else 0
            )
        
        print("\nDQL Exit Distribution Statistics:")
        exit_stats = dql_results_df.agg({
            f'exit_{i}_pct': ['mean', 'std'] for i in range(4)
        })
        print(exit_stats)
        
        # Generate comparison visualization
        print("\nGenerating DQL comparison visualization...")
        plot_dql_comparison(results_df, dql_results_df)
        print("DQL comparison visualization completed")
        
        # Generate visualizations with DQL overlay
        print("\nGenerating visualization with DQL comparison...")
        plot_stacked_3d_surfaces_side_by_side(results_df, dql_results_df)

        plot_logit_confidence_impact(results_df, dql_results_df)

    else:
        # Generate visualizations without DQL overlay
        print("\nGenerating visualization...")
        plot_stacked_3d_surfaces_side_by_side(results_df)
    
    print("Visualization completed")
    
    # Print best configurations
    print("\nBest configurations:")
    print("\nTop 5 by accuracy:")
    print(results_df.nlargest(5, 'accuracy')[
        ['threshold1', 'threshold2', 'threshold3', 'accuracy', 'avg_latency']
    ])
    
    print("\nTop 5 by latency (lowest):")
    print(results_df.nsmallest(5, 'avg_latency')[
        ['threshold1', 'threshold2', 'threshold3', 'accuracy', 'avg_latency']
    ])
    
    # Add this after the existing visualizations
    print("Generating exit distribution visualization...")
    plot_exit_distribution(results_df)
    print("Exit distribution visualization completed")
    
    # Print exit distribution statistics
    print("\nExit Distribution Statistics:")
    exit_stats = results_df.agg({
        f'exit_{i}_pct': ['mean', 'std', 'min', 'max'] for i in range(4)
    })
    print(exit_stats)
    
    # Find configurations with most balanced exit distribution
    results_df['exit_std'] = results_df[[f'exit_{i}_pct' for i in range(4)]].std(axis=1)
    print("\nMost balanced exit distributions (lowest standard deviation):")
    print(results_df.nsmallest(5, 'exit_std')[
        ['threshold1', 'threshold2', 'threshold3', 'accuracy', 'avg_latency'] + 
        [f'exit_{i}_pct' for i in range(4)] +
        ['exit_std']
    ])

def plot_previous_results():
    """
    Load previous results and generate side-by-side accuracy surface plots for three threshold3 values,
    with latency contour lines overlaid.
    """
    print("Loading previous results...")
    
    # Load threshold results
    results_df = pd.read_csv('models/visualwakewords/mobilenetv2/threshold_results_visualwakewords_mobilenetv2.csv')
    
    # Create 3D plots
    fig = plt.figure(figsize=(20, 8))
    
    # Get unique threshold values and filter out values below 0.7
    thresh1 = sorted([t for t in results_df['threshold1'].unique() if t >= 0.7])
    thresh2 = sorted([t for t in results_df['threshold2'].unique() if t >= 0.7])
    thresh3_values = sorted(results_df['threshold3'].unique())
    # Select three evenly spaced threshold3 values
    selected_thresh3 = [
        thresh3_values[0],  # First value
        thresh3_values[len(thresh3_values)//2],  # Middle value
        thresh3_values[-1]  # Last value
    ]
    
    # Create coordinate meshgrid
    X, Y = np.meshgrid(thresh1, thresh2)
    
    # Plot surface for each selected threshold3 value
    for idx, thresh3 in enumerate(selected_thresh3):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        
        Z_accuracy = np.zeros_like(X)
        Z_latency = np.zeros_like(X)
        for i, t1 in enumerate(thresh1):
            for j, t2 in enumerate(thresh2):
                mask = (results_df['threshold1'] == t1) & (results_df['threshold2'] == t2) & (results_df['threshold3'] == thresh3)
                if mask.any():
                    Z_accuracy[j,i] = results_df[mask]['accuracy'].values[0]
                    Z_latency[j,i] = results_df[mask]['avg_latency'].values[0]
                else:
                    print(f"Warning: Missing data for t1={t1}, t2={t2}, t3={thresh3}")
                    Z_accuracy[j,i] = np.nan
                    Z_latency[j,i] = np.nan
                    
        # Plot accuracy surface
        surf = ax.plot_surface(X, Y, Z_accuracy,
                             cmap='viridis',
                             alpha=0.8)
        
        # Add latency contours
        # Project onto z planes at regular intervals
        z_levels = np.linspace(Z_accuracy.min(), Z_accuracy.max(), 6)
        for z_level in z_levels:
            CS = ax.contour(X, Y, Z_latency, levels=10, 
                          zdir='z', offset=z_level,
                          colors='red', alpha=0.5, linewidths=1)
            ax.clabel(CS, inline=True, fontsize=8, fmt='%.2f')
            
        ax.set_xlabel('Exit 1 Threshold')
        ax.set_ylabel('Exit 2 Threshold')
        ax.set_zlabel('Accuracy')
        ax.view_init(elev=30, azim=160)
        ax.set_title(f'Threshold 3 = {thresh3:.2f}')
    
    plt.suptitle('Accuracy Surfaces with Latency Contours for Different Exit 3 Thresholds')
    plt.tight_layout()
    plt.savefig('threshold_accuracy_surfaces.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization completed")


if __name__ == "__main__":
    # main()
    plot_previous_results()
