"""
This program implements the Local Linear Embedding (LLE) algorithm to obtain, 
visualize, and evaluate low-dimensional representations of perovskite materials. 
Unlike the sparse and high-dimensional vectors, the embeddings produced by 
this method are dense and compact, capturing the local geometric structure 
of the data. By projecting the original material composition 
space into a lower-dimensional manifold, this approach facilitates both 
visualization and downstream machine learning tasks. The program includes routines 
for computing the embeddings, generating 2D and 3D plots, and quantitatively 
assessing the quality of the dimensionality reduction.
"""

def plot_tsne(points_data):
    """ Plots the t-SNE embeddings based on the provided point data.
    Args:
        points_data (dict): A dictionary where keys are conditions and values are
                             dictionaries containing 'names', 'x', 'y', and 'color'
                             lists for the selected points.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    for condition, data in points_data.items():
        plt.scatter(data["x"], data["y"], c=data["color"], s=10, label=condition)
        for xi, yi, label, color in zip(data["x"], data["y"], data["names"], [data["color"]] * len(data["names"])):
            plt.text(xi, yi, label, fontsize=7, alpha=0.7, color=color)

    plt.title('t-SNE Visualization of Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def pdf_of_embeddings(embeddings):
    # Number of dimensions
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    n_dims = embeddings.shape[1]
    # Determine subplot grid size (square or nearly square)
    n_cols = int(np.ceil(np.sqrt(n_dims)))
    n_rows = int(np.ceil(n_dims / n_cols))
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    # Loop over dimensions and plot
    for i in range(n_dims):
        data = embeddings[:, i]
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.min(), data.max() + 0.1, 500)
        pdf = kde(x_grid)

        axes[i].plot(x_grid, pdf, label=f'pdf of $L_{{{i+1}}}$', color='black')
        axes[i].plot(x_grid, pdf)
        axes[i].fill_between(x_grid, pdf, alpha=0.3)
        axes[i].set_xlabel(f'$L_{{{i+1}}}$')
        axes[i].set_ylabel('Probability')
        axes[i].legend()

    # Hide unused axes, if any
    for j in range(n_dims, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# **********************************
if __name__ == "__main__":

    import numpy as np
    import pandas as PD

    import sys
    sys.path.insert(0, './src')  # add src to Python path
    import LLE_utils
    from pathlib import Path
    # --- Input Parameters ---
    base_dir = Path(__file__).parent.parent    # project location
    file_name_data = base_dir / "data" / "Perovskite_database_content_all_data.csv"
    output_dim = 4   # The desired embedding dimension
    n_neighbors = 60
    # where the embedding will be saved for transforming the dataset. This part can be change by using
    # a few routines and the LLE model, as shown in the illustrative example of LLEs.
    embeddings_file = base_dir / "data" / 'embeddings_cosine.csv'
    basic_elements_file = base_dir / "data" / 'basic_elements.csv'
    type_embed = 'cosine'


    # Extracts and arranges the information required to obtain the raw vectors of perovskite materials.
    Data = LLE_utils.extract_ions_PCE(file_name_data)   # Information from dataset.
    elements_prvskt = LLE_utils.elementos_vocabulario(Data)
    print('The number of possible elements used to synthesize the perovskite material is:', len(elements_prvskt))
    print('The different elements used to synthesize the material are: ')
    print(elements_prvskt)
    df = PD.DataFrame(elements_prvskt, columns=['basic elements'])
    df.to_csv(basic_elements_file, index=False)
    
    name_prvkt, sparse_vectors = LLE_utils.sparse_vectors_prvskt(Data, elements_prvskt)    #sparse vectors of the whole dataset.
    
    # obtains the dictionary of sparse vectors for the 1903 possible perovskite materials.
    prvkt_dict = LLE_utils.sparse_dict_prvskt_materials(name_prvkt, sparse_vectors)
        
    # Once the raw representations are obtained, LLE transformation is estimated.
    embed_prvkt_dict = LLE_utils.LLE_transformation(prvkt_dict, type_embed, output_dim, n_neighbors, random_state = 42, method = 'modified')
    LLE_utils.from_dict2csv(embed_prvkt_dict, embeddings_file)
    name_prvkt = list(embed_prvkt_dict.keys())
    dense_vectors = list(embed_prvkt_dict.values())
    dense_vectors = np.array(dense_vectors)
    
    n_components = 2
    tsne_embeddings = LLE_utils.get_tsne_embeddings(dense_vectors, n_components=n_components)
    points_to_plot  = LLE_utils.get_points_to_plot(name_prvkt, tsne_embeddings)
    plot_tsne(points_to_plot)

    pdf_of_embeddings(dense_vectors)