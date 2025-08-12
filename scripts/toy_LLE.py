# toy example LLE. Se muestra el proceso de representación de algún compuesto en
# particular; luego, se muestra lo que sucede en la representación LLE cuando se
# cambia el contenido de Plomo. 




# # función tomada de LLE_prvskt_material.py
# def sparse_dict_prvskt_materials(name_prvkt, sparse_vectors):
# 	import numpy as np

# 	sparse_vectors = np.vstack(sparse_vectors)
# 	vector_sums = np.sum(sparse_vectors, axis=1, keepdims=True)  # Calculate the sum of each vector (sum along axis 1)
# 	normalized_data = sparse_vectors/vector_sums               # Normalize each vector by dividing by its sum
# 	normalized_data_list = normalized_data.tolist()              # Convert back to a list of lists if needed
# 	sparse_vectors = normalized_data_list
# 	prvkt_dict = dict()
# 	for name, vector in zip(name_prvkt, sparse_vectors):
# 		if name not in prvkt_dict:
# 			prvkt_dict[name] = vector

# 	return prvkt_dict                              # retorna la convesión en formato dictionario.



if __name__ == "__main__":

	import numpy as np
	import pandas as PD
	import sys
	sys.path.insert(0, './src')
	from pathlib import Path
	import LLE_utils

	# --- Input Parameters ---
	base_dir = Path(__file__).parent.parent    # project location
	basic_elements_file = base_dir / "data" / 'basic_elements.csv'
	df = PD.read_csv(basic_elements_file)
	basic_elements = df['basic elements'].tolist()
	print(basic_elements)

	material = ['Cs0.05FA0.61MA0.34PbBr0.45I2.55', 'Cs0.05FA0.66MA0.29PbBr0.45I2.55', 'Cs0.05FA0.71MA0.24PbBr0.45I2.55', 'Cs0.05FA0.76MA0.19PbBr0.45I2.55','Cs0.05FA0.81MA0.14PbBr0.45I2.55', 'Cs0.05FA0.85MA0.1PbBr0.45I2.55', 'Cs0.05FA0.9MA0.05PbBr0.45I2.55']
	vectores = []
	for mat in material:
		vector = LLE_utils.perovskite_to_vector(mat, basic_elements)
		vectores.append(vector)
	vectores = np.vstack(vectores)
	prvkt_dict = LLE_utils.sparse_dict_prvskt_materials(material, vectores)

	name_prvkt = list(prvkt_dict.keys())
	sparse_vectors = list(prvkt_dict.values())
	data_matrix = np.array(sparse_vectors)

	import joblib
	lle_cos = joblib.load('lle_cos_model.pkl')
	V_transf = lle_cos.transform(data_matrix)
	print(V_transf)
	#print(vectores)

	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

	# Example: a (N, 4) array
	vectors = V_transf  # Replace this with your actual data
	# Split components
	x = vectors[:, 0]
	y = vectors[:, 1]
	z = vectors[:, 2]
	c = vectors[:, 3]  # color dimension

	# 3D scatter plot
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111, projection='3d')

	p = ax.scatter(x, y, z, c=c, cmap='viridis', s=50)
	# Add colorbar for 4th dimension
	cb = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.02, location='left')
	cb.set_label(r'$L_4$', rotation=90, labelpad=10)

	# Add text labels
	for xi, yi, zi, name in zip(x, y, z, material):
		ax.text(xi, yi, zi + 0.00028, name, fontsize=11, color='black', backgroundcolor='white')

	ax.set_xlabel(r'$L_1$')
	ax.set_ylabel(r'$L_2$')
	ax.set_zlabel(r'$L_3$')
	plt.tight_layout()
	plt.show()
