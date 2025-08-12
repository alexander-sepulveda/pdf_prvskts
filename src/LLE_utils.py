def extract_ions_PCE(name_file_data):
    # takes data from the original Jacobson's dataset and extracts information about the perovskite components.
    import os
    import pandas as PD
    import ABX
    # var_names : name of varibles to be read from Perovskite dataset.
    var_names = [
            'Perovskite_composition_long_form',
            'Perovskite_composition_a_ions','Perovskite_composition_a_ions_coefficients',
            'Perovskite_composition_b_ions','Perovskite_composition_b_ions_coefficients',
            'Perovskite_composition_c_ions','Perovskite_composition_c_ions_coefficients',
            'Perovskite_composition_perovskite_ABC3_structure',
            'JV_default_PCE'            
            ]

    Data = PD.read_csv(name_file_data, low_memory=False)
    Data = Data[var_names]
    
    # 1. we eliminate entries corresponding to non-sigle layer solar cells in perovskite layer. 
    #    They are recognized by having a vertical bar "|" 
    Data = ABX.Remove_nonsingle_layer(Data)

    # Eliminate those rows having "NaN" in text format.
    Data = Data.dropna()
    Data = Data[~Data['Perovskite_composition_a_ions'].str.contains("NaN", na=False)]
    Data = Data[~Data['Perovskite_composition_b_ions'].str.contains("NaN", na=False)]
    Data = Data[~Data['Perovskite_composition_c_ions'].str.contains("NaN", na=False)]
    # Remove rows where any column contains 'x'. It means no information.
    Data = Data[~Data.apply(lambda row: row.astype(str).str.contains('x', case=False)).any(axis=1)]
    Data = Data.reset_index(drop=True)  # removes the old index instead of adding it as a new column

    return Data

def elementos_vocabulario(Data):
    # Obtains the vectors representing the different perovskite materials appearing in the dataset.
    mol_A = associate_elements_with_numbers(Data['Perovskite_composition_a_ions'], Data['Perovskite_composition_a_ions_coefficients'])
    mol_B = associate_elements_with_numbers(Data['Perovskite_composition_b_ions'], Data['Perovskite_composition_b_ions_coefficients'])
    mol_X = associate_elements_with_numbers(Data['Perovskite_composition_c_ions'], Data['Perovskite_composition_c_ions_coefficients'])
    # Usar un solo conjunto para almacenar todos los elementos únicos
    all_elements = set()

    def add_elements(mol_list):
        for dictionary in mol_list:
            if dictionary is not None:  # Verificar si no es None
                all_elements.update(dictionary.keys())

    # Agregar elementos desde cada conjunto mol_A, mol_B y mol_X
    add_elements(mol_A)
    add_elements(mol_B)
    add_elements(mol_X)
    # Convertir el conjunto en una lista ordenada
    all_elements_list = sorted(list(all_elements))

    return all_elements_list  # Returns the list of unique elements

def associate_elements_with_numbers(elements, numbers):
    # The dataSet has the ion information in separate columns, both in text format.
    # This function joins the information from both columns.
    associations = []
    
    for elem_list, num_list in zip(elements, numbers):
        elem_split = elem_list.split('; ')
        num_split = num_list.split('; ')
        
        if len(elem_split) == len(num_split):  # Ensure they match in length
            associations.append(dict(zip(elem_split, map(float, num_split))))
        else:
            associations.append(None)  # Handle mismatches
    
    return associations

def sparse_vectors_prvskt(Data, all_elements_list):
    # The dataset contains ion information in two separate columns, both in text format.
    # This program combines the information from both columns.
    mol_A = associate_elements_with_numbers(Data['Perovskite_composition_a_ions'], Data['Perovskite_composition_a_ions_coefficients'])
    none_indices_A = find_None_values(mol_A)  # debe hacerse porque no siempre puede hacer la conversión.
    mol_B = associate_elements_with_numbers(Data['Perovskite_composition_b_ions'], Data['Perovskite_composition_b_ions_coefficients'])
    none_indices_B = find_None_values(mol_B)
    mol_X = associate_elements_with_numbers(Data['Perovskite_composition_c_ions'], Data['Perovskite_composition_c_ions_coefficients'])
    none_indices_X = find_None_values(mol_X)
    none_indices = none_indices_A + none_indices_B + none_indices_X
    
    # Remove the elements at the specified indexes
    mol_A = [d for idx, d in enumerate(mol_A) if idx not in none_indices]
    mol_B = [d for idx, d in enumerate(mol_B) if idx not in none_indices]
    mol_X = [d for idx, d in enumerate(mol_X) if idx not in none_indices]
    Data = Data.drop(none_indices).reset_index(drop=True)

    # Generate vectors
    vectors_A = one_hot_and_quantity_vectors(mol_A, all_elements_list)
    vectors_B = one_hot_and_quantity_vectors(mol_B, all_elements_list)
    vectors_X = one_hot_and_quantity_vectors(mol_X, all_elements_list)
    summed_vectors = [a + b + x for a, b, x in zip(vectors_A, vectors_B, vectors_X)]   # this vector represents the whole perovskite material.

    name_prvkt = Data['Perovskite_composition_long_form'].tolist()
    if len(name_prvkt) != len(summed_vectors):
        print('ERROR: lengths of vectors in sparse_vectors_prvskt() are different.')

    return name_prvkt, summed_vectors

def one_hot_and_quantity_vectors(mol, all_elements_list):
    # Function to generate one-hot + quantity vector
    # Create a mapping of element → index
    import numpy as np
    element_to_index = {element: i for i, element in enumerate(all_elements_list)}
    num_elements = len(element_to_index)

    the_vectors = []
    for molecule in mol:
        hot_vector = np.zeros(num_elements, dtype=float)   # Initialize vector (zeros)
        componentes = list(molecule.keys())
        N_componentes = len(componentes)
        for ii in range(0, N_componentes):
            index = all_elements_list.index(componentes[ii])
            hot_vector[index] = molecule[componentes[ii]]

        the_vectors.append(hot_vector)

    return np.array(the_vectors)

def find_None_values(mol):  
    none_indices = []
    for idx, d in enumerate(mol):
        if d is None:
            none_indices.append(idx)

    return(none_indices)

def sparse_dict_prvskt_materials(name_prvkt, sparse_vectors):
    import numpy as np

    vector_sums = np.sum(sparse_vectors, axis=1, keepdims=True)  # Calculate the sum of each vector (sum along axis 1)
    normalized_data = sparse_vectors/vector_sums                 # Normalize each vector by dividing by its sum
    normalized_data_list = normalized_data.tolist()              # Convert back to a list of lists if needed
    sparse_vectors = normalized_data_list
    prvkt_dict = dict()
    for name, vector in zip(name_prvkt, sparse_vectors):
        if name not in prvkt_dict:
            prvkt_dict[name] = vector

    return prvkt_dict

def from_dict2csv(embeddings_dict, name_file_embeddings):
    import csv
    
    with open(name_file_embeddings, mode="w", newline='') as file:
        writer = csv.writer(file)
        # Write header
        D = len(next(iter(embeddings_dict.values())))
        header = ['Composite'] + [f'LLE_{i+1}' for i in range(D)]
        writer.writerow(header)

        # Write each row
        for key, vec in embeddings_dict.items():
            writer.writerow([key] + vec.tolist())

def LLE_transformation(prvkt_dict, type_embed, n_components: int, n_neighbors: int, random_state: int = 42, method: str = 'modified'):
    """ Performs Locally Linear Embedding (LLE) for dimensionality reduction.
    Args:
        sparse_vectors (List[np.ndarray]): List of sparse input vectors (each of shape (142,)).
        n_components (int): The desired number of dimensions for the output embeddings.
        n_neighbors (int): The number of neighbors to consider for each point.
        rand_state (int, optional): Random state for reproducibility. Defaults to 42.
        method (str, optional): LLE method to use. Defaults to 'standard'.
    Returns:
        numpy.ndarray: The LLE embeddings (shape: (number_of_vectors, n_components)).
    """
    # Convert the list of numpy arrays to a 2D numpy array

    from sklearn.manifold import LocallyLinearEmbedding
    import numpy as np

    name_prvkt = list(prvkt_dict.keys())
    sparse_vectors = list(prvkt_dict.values())
    data_matrix = np.array(sparse_vectors)
    if type_embed=='euclidean':
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state, method=method) # Initialize the LLE model
        lle_embeddings = lle.fit_transform(data_matrix)     # Perform LLE
    elif type_embed=='cosine':
        lle_embeddings = cosine_LLE(data_matrix, n_components, n_neighbors, method)
    else:
        print('ERROR especfying type of distance for embedding.')

    embed_prvkt_dict = dict()
    for name, vector in zip(name_prvkt, lle_embeddings):
        if name not in embed_prvkt_dict:
            embed_prvkt_dict[name] = vector

    from sklearn.manifold import trustworthiness
    score = trustworthiness(sparse_vectors, lle_embeddings)
    print('*********')
    print("Trustworthiness of LLE embeddings:", score)

    return embed_prvkt_dict

def cosine_LLE(data_matrix, n_components, n_neighbors, method):
    import joblib
    from sklearn.manifold import LocallyLinearEmbedding
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    X = data_matrix  # shape (n_samples, n_features)
    # Parameters
    random_state = 42
    # Step 1: Compute cosine distances
    #cosine_dists = cosine_distances(X)
    # Step 2: Find the k-nearest neighbors based on cosine distance
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    nbrs.fit(X)
    neighbors_indices = nbrs.kneighbors(return_distance=False)
    # Step 3: Use the neighborhood graph in LLE
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, method='standard',neighbors_algorithm='brute',random_state=random_state)
    X_lle = lle.fit_transform(X)
    joblib.dump(lle, 'lle_cos_model.pkl')
    return X_lle

def perovskite_to_vector(material_str, basic_elements):
    import numpy as np
    import re

    vec = np.zeros(len(basic_elements))
    i = 0
    while i < len(material_str):
        match = None
        for j in range(len(material_str), i, -1):
            prefix = material_str[i:j]
            if prefix in basic_elements:
                match = prefix
                break
        if match:
            i += len(match)
            num_match = re.match(r'(\d*\.?\d*)', material_str[i:])
            if num_match:
                num_str = num_match.group(1)
                quantity = float(num_str) if num_str else 1.0
                i += len(num_str)
            else:
                quantity = 1.0
            idx = basic_elements.index(match)
            vec[idx] += quantity
        else:
            # Unrecognized token — skip one character to avoid infinite loop
            i += 1

    return vec






# ------------ plot t-SNE ------------
def get_tsne_embeddings(lle_embeddings, n_components=2, perplexity=30, random_state=42):
    """  Generates t-SNE embeddings from the given LLE embeddings.
    Returns a np.ndarray: The t-SNE embeddings. """
    from sklearn.manifold import TSNE
    import numpy as np

    tsne = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
    tsne_embeddings = tsne.fit_transform(lle_embeddings)
    return tsne_embeddings

def get_points_to_plot(name_prvkt, tsne_embeddings):
    """ Selects points and their t-SNE coordinates based on specified conditions.
    Args: 1) name_prvkt (list): List of perovskite names.
          2) tsne_embeddings (np.ndarray): The t-SNE embeddings.
    Returns a dictionary where keys are conditions and values are dictionaries containing 'names', 'x', and 'y' lists for the selected points.
    """
    name_prvkt, tsne_embeddings = randomly_sample_data(name_prvkt, tsne_embeddings, sample_fraction=0.4, random_seed=42)

    points_to_plot = {
        "MA.Pb.I": {"names": [], "x": [], "y": [], "color": 'red'},
        "FA.Pb.I": {"names": [], "x": [], "y": [], "color": 'blue'},
        "Cs.MA.FA.Pb.I": {"names": [], "x": [], "y": [], "color": 'green'},
        "no-MA.no-FA.Cs": {"names": [], "x": [], "y": [], "color": 'black'},
    }

    for element, v, w in zip(name_prvkt, tsne_embeddings[:, 0], tsne_embeddings[:, 1]):
        if element.startswith("MA") and "Pb" in element and "I" in element and "FA" not in element and "Cs" not in element:
            points_to_plot["MA.Pb.I"]["names"].append(element)
            points_to_plot["MA.Pb.I"]["x"].append(v)
            points_to_plot["MA.Pb.I"]["y"].append(w)
        elif element.startswith("FA") and "Pb" in element and "I" in element and "MA" not in element and "Cs" not in element:
            points_to_plot["FA.Pb.I"]["names"].append(element)
            points_to_plot["FA.Pb.I"]["x"].append(v)
            points_to_plot["FA.Pb.I"]["y"].append(w)
        elif element.startswith("Cs") and "MA" in element and "FA" in element and "Pb" in element and "I" in element:
            points_to_plot["Cs.MA.FA.Pb.I"]["names"].append(element)
            points_to_plot["Cs.MA.FA.Pb.I"]["x"].append(v)
            points_to_plot["Cs.MA.FA.Pb.I"]["y"].append(w)
        elif "Cs" in element and "MA" not in element and "FA" not in element:
            points_to_plot["no-MA.no-FA.Cs"]["names"].append(element)
            points_to_plot["no-MA.no-FA.Cs"]["x"].append(v)
            points_to_plot["no-MA.no-FA.Cs"]["y"].append(w)

    return points_to_plot

def randomly_sample_data(name_prvkt, tsne_embeddings, sample_fraction, random_seed=None):
    # Randomly samples a fraction of the perovskite names and their corresponding t-SNE embeddings.
    import random
    import numpy as np

    if len(name_prvkt) != tsne_embeddings.shape[0]:
        raise ValueError("The length of name_prvkt must match the number of rows in tsne_embeddings.")

    if not (0 <= sample_fraction <= 1):
        raise ValueError("sample_fraction must be between 0 and 1.")

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed) # Also seed numpy for consistency

    num_samples = int(len(name_prvkt) * sample_fraction)
    indices = random.sample(range(len(name_prvkt)), num_samples)
    indices.sort() # Sort indices to maintain order (optional, but can be helpful)

    sampled_name_prvkt = [name_prvkt[i] for i in indices]
    sampled_tsne_embeddings = tsne_embeddings[indices]

    return sampled_name_prvkt, sampled_tsne_embeddings

