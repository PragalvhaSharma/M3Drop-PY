import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib_venn import venn3, venn3_circles

def bg__dropout_plot_base(expr_mat, xlim=None, suppress_plot=False):
    """
    Creates the base for dropout rate vs. expression plots.
    """
    from .utils import bg__calc_variables
    
    gene_info = bg__calc_variables(expr_mat)
    
    xes = np.log10(gene_info['s'].replace(0, 1e-10))
    p = gene_info['p']
    
    if not suppress_plot:
        # Density scatter plot
        xy = np.vstack([xes,p])
        z = gaussian_kde(xy)(xy)
        
        fig, ax = plt.subplots()
        ax.scatter(xes, p, c=z, s=10, cmap='viridis')
        
        if xlim:
            ax.set_xlim(xlim)
        ax.set_ylim(0, 1)
        ax.set_xlabel("log10(expression)")
        ax.set_ylabel("Dropout Rate")
        
    return {'gene_info': gene_info, 'xes': xes, 'order': np.argsort(xes)}


def bg__add_model_to_plot(fitted_model, base_plot, lty='-', lwd=1, col="dodgerblue", legend_loc="top right"):
    """
    Adds a model fit to a dropout plot.
    """
    plt.plot(
        base_plot['xes'][base_plot['order']],
        fitted_model['predictions'][base_plot['order']],
        linestyle=lty,
        linewidth=lwd,
        color=col,
        label=fitted_model['model']
    )
    if legend_loc:
        plt.legend(loc=legend_loc)


def bg__highlight_genes(base_plot, expr_mat, genes, col="darkorange", pch=16):
    """
    Highlights specific genes on a plot.
    """
    if not isinstance(genes, (list, np.ndarray, pd.Series)):
        genes = [genes]
        
    if isinstance(expr_mat, pd.DataFrame):
        gene_indices = [expr_mat.index.get_loc(g) for g in genes if g in expr_mat.index]
    else:
        gene_indices = genes # Assume integer indices
        
    if len(gene_indices) > 0:
        plt.scatter(
            base_plot['xes'][gene_indices],
            base_plot['gene_info']['p'][gene_indices],
            c=col,
            s=50, # bigger size to highlight
            marker='o' if pch==16 else 'x'
        )


def bg__expression_heatmap(genes, expr_mat, cell_labels=None, gene_labels=None, key_genes=None, key_cells=None):
    """
    Internal function to plot a customized heatmap of scaled log expression values.
    """
    if not isinstance(genes, (list, np.ndarray, pd.Series)):
        genes = [genes]
        
    if isinstance(expr_mat, pd.DataFrame):
        heat_data = expr_mat.loc[genes]
    else:
        heat_data = expr_mat[genes,:]
    
    heat_data = np.log1p(heat_data) / np.log(2)
    
    # Create colormap for cell labels
    if cell_labels is not None:
        unique_labels = np.unique(cell_labels)
        lut = dict(zip(unique_labels, sns.color_palette("Set3", len(unique_labels))))
        col_colors = pd.Series(cell_labels, index=heat_data.columns).map(lut)
    else:
        col_colors = None
        
    # Create colormap for gene labels
    if gene_labels is not None:
        unique_glabels = np.unique(gene_labels)
        glut = dict(zip(unique_glabels, sns.color_palette("Set1", len(unique_glabels))))
        row_colors = pd.Series(gene_labels, index=heat_data.index).map(glut)
    else:
        row_colors = None
        
    g = sns.clustermap(
        heat_data,
        z_score=0, # scale="row" in R
        cmap="RdBu_r", # rev(brewer.pal(11,"RdBu"))
        col_colors=col_colors,
        row_colors=row_colors,
        linewidths=0,
        yticklabels=key_genes is not None,
        xticklabels=key_cells is not None,
        vmin=-2, vmax=2 # breaks
    )
    
    if key_genes is not None:
        g.ax_heatmap.set_yticklabels([label.get_text() if label.get_text() in key_genes else "" for label in g.ax_heatmap.get_yticklabels()])

    if key_cells is not None:
        g.ax_heatmap.set_xticklabels([label.get_text() if label.get_text() in key_cells else "" for label in g.ax_heatmap.get_xticklabels()])

    return g


def M3DropExpressionHeatmap(genes, expr_mat, cell_labels=None, interesting_genes=None, key_genes=None, key_cells=None):
    """
    Plots a customized heatmap of gene expression.
    """
    if key_genes is None:
        key_genes = genes
    
    gene_labels = None
    if interesting_genes is not None:
        if isinstance(interesting_genes, list) and all(isinstance(sublist, list) for sublist in interesting_genes):
             gene_labels = pd.Series(0, index=genes)
             for i, sublist in enumerate(interesting_genes):
                 gene_labels[gene_labels.index.isin(sublist)] = i + 1
        else:
             gene_labels = pd.Series(genes.isin(interesting_genes).astype(int), index=genes)
    
    heatmap_output = bg__expression_heatmap(genes, expr_mat, cell_labels=cell_labels, gene_labels=gene_labels, key_genes=key_genes, key_cells=key_cells)
    
    return heatmap_output


def M3DropThreeSetVenn(sets, set_names, ax=None, **kwargs):
    """
    Creates a three-way proportional-area Venn diagram.

    This function uses the `matplotlib-venn` library to create a Venn
    diagram for three sets. The areas of the circles and overlaps are
    proportional to the number of elements in each set and intersection.

    Parameters:
    - sets (list of set): A list containing three sets of elements.
    - set_names (list of str): A list of three names for the sets.
    - ax (matplotlib.axes.Axes, optional): The axes to plot on. If None,
      a new figure and axes are created.
    - **kwargs: Additional keyword arguments passed to `venn3`.

    Returns:
    - matplotlib_venn.VennDiagram: The Venn diagram object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    v = venn3(sets, set_names, ax=ax, **kwargs)

    # Proportional circles
    venn3_circles(subsets=[len(s) for s in sets], ax=ax, linestyle="solid")

    ax.set_title("Proportional Venn Diagram")
    ax.axis('on')
    plt.show()

    return v
