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

# --------------------
def pdf_of_embeddings(embeddings):
    # Number of dimensions
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    n_dims = embeddings.shape[1]
    # Determine subplot grid size (square or nearly square)
    n_cols = int(np.ceil(np.sqrt(n_dims)))
    n_rows = int(np.ceil(n_dims / n_cols))
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

    for j in range(n_dims, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


# ----------------------
def crossval_LLE(prvkt_dict):
    import numpy as np
    import matplotlib.pyplot as plt
    import itertools
    import LLE_utils # Assuming your functions are in this module

    # --- Hyperparameters ---
    LLE_COMPONENTS = 4
    LLE_NEIGHBORS = 60 # A robust choice, adjust if needed
    TSNE_COMPONENTS = 2
    TYPE_EMBED = 'cosine' # Specify your embed type

    n_splits = 5
    all_keys = list(prvkt_dict.keys())
    np.random.seed(82)
    np.random.shuffle(all_keys)   # Shuffle keys for a random split

    # Split the list of keys into 5 folds
    key_folds = np.array_split(all_keys, n_splits)
    print(f"Total material groups: {len(all_keys)}. Splitting into {n_splits} folds.")

    fig, axes = plt.subplots(3, 2, figsize=(13, 13), sharex=True, sharey=True) 
    fig.suptitle(f"t-SNE Visualization of LLE (n={LLE_COMPONENTS}) structure across 5 CV folds", fontsize=14)
    print(f"Running {n_splits}-fold CV...")
    axes_iter = axes.flat

    for i in range(n_splits):
        ax = axes.flat[i] 
        # Get the keys for this fold's training set
        train_keys = list(itertools.chain.from_iterable(key_folds[:i] + key_folds[i+1:]))
        train_dict = {key: prvkt_dict[key] for key in train_keys}   # Create the new dictionary containing only the training data
        print(f"  Fold {i+1}: Training with {len(train_keys)} groups...")

        # --- Call my LLE function ---
        embed_prvkt_dict = LLE_utils.LLE_transformation(train_dict, TYPE_EMBED, LLE_COMPONENTS, LLE_NEIGHBORS, random_state=42, method='modified')
        name_prvkt = list(embed_prvkt_dict.keys())
        dense_vectors = list(embed_prvkt_dict.values())
        dense_vectors = np.array(dense_vectors)
        n_components = 2
        tsne_embeddings = LLE_utils.get_tsne_embeddings(dense_vectors, n_components=n_components)
        points_to_plot  = LLE_utils.get_points_to_plot(name_prvkt, tsne_embeddings)

        ax = axes_iter[i] 
        for condition, data in points_to_plot.items():
            ax.scatter(data["x"], data["y"], c=data["color"], s=10, label=condition) 

        for xi, yi, label, color in zip(data["x"], data["y"], data["names"], [data["color"]] * len(data["names"])):
            ax.text(xi, yi, label, fontsize=7, alpha=0.7, color=color) 
            ax.set_title(f'Fold {i+1} t-SNE Visualization') # CHANGED to ax.set_title
            ax.set_xlabel('t-SNE Dim 1')
            ax.set_ylabel('t-SNE Dim 2')
            ax.grid(True)

    handles, labels = ax.get_legend_handles_labels() # Get labels from the last axis
    fig.legend(handles, labels, loc='upper right', title="Material Groups")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    
    # obtains the dictionary of sparse vectors for the different perovskite materials in the dataset.
    prvkt_dict = LLE_utils.sparse_dict_prvskt_materials(name_prvkt, sparse_vectors)

    # -- it is used for selecting output_dim and n_neighbors
    crossval_LLE(prvkt_dict)

    
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