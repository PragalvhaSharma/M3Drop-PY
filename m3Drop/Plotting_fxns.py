import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist
from matplotlib.colors import LinearSegmentedColormap
import warnings
from .basics import bg__calc_variables


def bg__dropout_plot_base(expr_mat, xlim=None, suppress_plot=False):
    """Create base dropout plot."""
    gene_info = bg__calc_variables(expr_mat)
    
    xes = np.log10(gene_info['s'])
    put_in_order = np.argsort(xes)
    
    if not suppress_plot:
        plt.figure(figsize=(8, 6))
        
        # Create density colors (approximating R's densCols)
        from matplotlib.colors import to_hex
        colors = plt.cm.Greys(np.linspace(0.3, 1.0, len(xes)))
        
        if xlim is not None:
            plt.scatter(xes, gene_info['p'], c=colors, s=16, alpha=0.7)
            plt.xlim(xlim)
        else:
            plt.scatter(xes, gene_info['p'], c=colors, s=16, alpha=0.7)
        
        plt.ylim(0, 1)
        plt.ylabel('Dropout Rate')
        plt.xlabel('log10(expression)')
        plt.title('')
    
    return {
        'gene_info': gene_info,
        'xes': xes,
        'order': put_in_order
    }


def bg__add_model_to_plot(fitted_model, base_plot, lty='-', lwd=1, col='dodgerblue', legend_loc='upper right'):
    """Add model line to existing plot."""
    ordered_xes = base_plot['xes'][base_plot['order']]
    ordered_predictions = fitted_model['predictions'][base_plot['order']]
    
    plt.plot(ordered_xes, ordered_predictions, linestyle=lty, linewidth=lwd, color=col)
    
    if legend_loc:
        plt.legend([fitted_model['model']], loc=legend_loc)


def bg__highlight_genes(base_plot, expr_mat, genes, col='darkorange', marker='o'):
    """Highlight specific genes on the plot."""
    if not isinstance(genes, (list, np.ndarray)) or not np.issubdtype(type(genes[0]), np.number):
        # Convert gene names to indices
        if hasattr(expr_mat, 'index'):
            gene_indices = [expr_mat.index.get_loc(gene) for gene in genes if gene in expr_mat.index]
        else:
            gene_indices = [i for i, row_name in enumerate(expr_mat) if row_name in genes]
        
        nomatch = len(genes) - len(gene_indices)
        if nomatch > 0:
            warnings.warn(f"{nomatch} genes could not be matched to data, they will not be highlighted.")
        
        if len(gene_indices) == 0:
            return np.array([[np.nan, np.nan], [np.nan, np.nan]])
        
        genes = gene_indices
    
    highlighted_x = base_plot['xes'][genes]
    highlighted_y = base_plot['gene_info']['p'][genes]
    
    plt.scatter(highlighted_x, highlighted_y, c=col, marker=marker, s=50, zorder=5)
    
    return np.column_stack([base_plot['gene_info']['s'][genes], highlighted_y])


def bg__expression_heatmap(genes, expr_mat, cell_labels=None, gene_labels=None, key_genes=None, key_cells=None):
    """Create expression heatmap."""
    if not isinstance(genes, (list, np.ndarray)) or not np.issubdtype(type(genes[0]), np.number):
        if hasattr(expr_mat, 'index'):
            new_genes = [expr_mat.index.get_loc(gene) for gene in genes if gene in expr_mat.index]
        else:
            new_genes = [i for i, row_name in enumerate(expr_mat) if row_name in genes]
        
        nomatch = len(genes) - len(new_genes)
        if nomatch > 0:
            warnings.warn(f"Warning: {nomatch} genes could not be matched to data, they will not be included in the heatmap.")
        
        genes = new_genes
    
    if len(genes) < 1:
        raise ValueError("Error: No genes for heatmap.")
    
    # Prepare data
    if isinstance(expr_mat, pd.DataFrame):
        heat_data = expr_mat.iloc[genes, :].values
    else:
        heat_data = expr_mat[genes, :]
    
    heat_data = np.log2(heat_data + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colormap (reverse RdBu)
    colors = ['#67001F', '#B2182B', '#D6604D', '#F4A582', '#FDDBC7', 
              '#FFFFFF', '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061']
    cmap = LinearSegmentedColormap.from_list('RdBu_r', colors[::-1])
    
    # Create heatmap
    im = ax.imshow(heat_data, cmap=cmap, aspect='auto', vmin=-2, vmax=2)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Z-Score', rotation=270, labelpad=15)
    
    # Set labels
    if key_genes is not None:
        row_labels = [''] * len(genes)
        for i, gene_idx in enumerate(genes):
            if hasattr(expr_mat, 'index'):
                gene_name = expr_mat.index[gene_idx]
            else:
                gene_name = str(gene_idx)
            if gene_name in key_genes:
                row_labels[i] = gene_name
        ax.set_yticklabels(row_labels)
    
    if key_cells is not None:
        col_labels = [''] * heat_data.shape[1]
        for i in range(heat_data.shape[1]):
            if hasattr(expr_mat, 'columns'):
                cell_name = expr_mat.columns[i]
            else:
                cell_name = str(i)
            if cell_name in key_cells:
                col_labels[i] = cell_name
        ax.set_xticklabels(col_labels, rotation=90)
    
    # Add side colors for cell labels
    if cell_labels is not None:
        unique_labels = list(set(cell_labels))
        colors_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        label_to_color = dict(zip(unique_labels, colors_palette))
        
        # Create legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=label_to_color[label]) 
                          for label in unique_labels]
        ax.legend(legend_elements, unique_labels, loc='center left', bbox_to_anchor=(1.1, 0.5))
    
    plt.tight_layout()
    
    # Perform clustering
    if heat_data.shape[0] < 10000:
        # Cluster both rows and columns
        row_linkage = linkage(pdist(heat_data), method='ward')
        col_linkage = linkage(pdist(heat_data.T), method='ward')
        
        return {
            'heatmap': im,
            'row_linkage': row_linkage,
            'col_linkage': col_linkage,
            'data': heat_data
        }
    else:
        # Only cluster columns for large datasets
        col_linkage = linkage(pdist(heat_data.T), method='ward')
        
        return {
            'heatmap': im,
            'row_linkage': None,
            'col_linkage': col_linkage,
            'data': heat_data
        }


def M3DropExpressionHeatmap(genes, expr_mat, cell_labels=None, interesting_genes=None, key_genes=None, key_cells=None):
    """Create M3Drop expression heatmap with gene labels."""
    # Convert known DE genes into heatmap labels
    gene_labels = np.ones(len(genes))
    
    if interesting_genes is not None:
        if isinstance(interesting_genes, list) and all(isinstance(x, list) for x in interesting_genes):
            # List of lists
            for i, gene_group in enumerate(interesting_genes):
                mask = np.isin(genes, gene_group)
                gene_labels[mask] = i + 2
        else:
            # Single list
            mask = np.isin(genes, interesting_genes)
            gene_labels[mask] = 2
    else:
        gene_labels = None
    
    # Handle key_genes and key_cells conversion
    if isinstance(key_genes, (list, np.ndarray)) and np.issubdtype(type(key_genes[0]), np.number):
        if hasattr(expr_mat, 'index'):
            key_genes = expr_mat.index[key_genes].tolist()
        else:
            key_genes = [str(i) for i in key_genes]
    
    if isinstance(key_cells, (list, np.ndarray)) and np.issubdtype(type(key_cells[0]), np.number):
        if hasattr(expr_mat, 'columns'):
            key_cells = expr_mat.columns[key_cells].tolist()
        else:
            key_cells = [str(i) for i in key_cells]
    
    # Handle genes input format
    if hasattr(genes, 'dtype') and genes.dtype.name == 'category':
        genes = genes.astype(str).tolist()
    
    if isinstance(genes, pd.DataFrame):
        # Find gene column
        gene_cols = [col for col in genes.columns if 'gene' in col.lower()]
        if len(gene_cols) == 1:
            genes = genes[gene_cols[0]].tolist()
        else:
            raise ValueError("Error: please provide a vector of gene names not a table.")
    
    heatmap_output = bg__expression_heatmap(
        genes, expr_mat, 
        cell_labels=cell_labels, 
        gene_labels=gene_labels, 
        key_genes=key_genes, 
        key_cells=key_cells
    )
    
    return heatmap_output


def M3DropGetHeatmapClusters(heatout, k, cluster_type="cell"):
    """Extract clusters from heatmap dendrogram."""
    if "gene" in cluster_type.lower() or "row" in cluster_type.lower():
        if heatout['row_linkage'] is None:
            raise ValueError("Row clustering not available for this heatmap")
        linkage_matrix = heatout['row_linkage']
        n_items = heatout['data'].shape[0]
    elif "cell" in cluster_type.lower() or "col" in cluster_type.lower():
        linkage_matrix = heatout['col_linkage']
        n_items = heatout['data'].shape[1]
    else:
        raise ValueError("cluster_type must contain 'gene', 'row', 'cell', or 'col'")
    
    # Cut tree to get k clusters
    clusters = cut_tree(linkage_matrix, n_clusters=k).flatten()
    
    # Create groups array (1-indexed like R)
    groups = clusters + 1
    
    return groups


def M3DropGetHeatmapNames(heatout, name_type="cell"):
    """Get names from heatmap."""
    if "gene" in name_type.lower() or "row" in name_type.lower():
        if hasattr(heatout['data'], 'index'):
            return heatout['data'].index.tolist()
        else:
            return list(range(heatout['data'].shape[0]))
    elif "cell" in name_type.lower() or "col" in name_type.lower():
        if hasattr(heatout['data'], 'columns'):
            return heatout['data'].columns.tolist()
        else:
            return list(range(heatout['data'].shape[1]))
    else:
        raise ValueError("name_type must contain 'gene', 'row', 'cell', or 'col'")
