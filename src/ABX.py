def read_prvskt_data():
    from pathlib import Path
    import pandas as PD
    # --- Input Parameters ---
    base_dir = Path(__file__).parent.parent    # project location
    file_name_data = base_dir / "data" / "Descriptors_Perovskite_Database.xlsx"

    Data = PD.read_excel(file_name_data)
    var_names = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'first_Prvskt_annealing_temperature',
    'first_Prvskt_thermal_annealing_time', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap',
    'Perovskite_thickness', 'Cell_area_measured', 'JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE']
    # filtrado de datos.
    Data = Data[var_names]
    Data = Data.dropna()
    Data = Data[Data['JV_default_PCE'] > 10]      # elimina valores visualmente raros
    Data = Data[Data['JV_default_Voc'] > 0.6]     # elimina valores visualmente raros
    Data = Data[Data['Perovskite_band_gap'] > 1.1]     # elimina valores visualmente raros
    Data = Data[Data['JV_default_Jsc'] > 10]      # elimina valores visualmente raros
    Data = Data[Data['Cell_area_measured'] < 5]      # elimina valores visualmente raros
    Data = Data.drop_duplicates()
    var_inputs = ['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4', 'DMF_DMSO_ratio', 'Perovskite_annealing_thermal_exposure', 'Perovskite_band_gap', 'first_Prvskt_annealing_temperature', 'Cell_area_measured']
    var_outputs = ['JV_default_PCE']

    return Data



def Remove_Nonsingle_Layers(Data):
    # only those with the ABX structure.
    Data.loc[:, 'Perovskite_composition_perovskite_ABC3_structure'] = (Data['Perovskite_composition_perovskite_ABC3_structure'].astype(str).str.upper() == 'TRUE')
    Data = Data[Data['Perovskite_composition_perovskite_ABC3_structure'] == True]
    # Eliminar filas con "|" en la columna 'Perovskite_composition_long_form'
    var_names_para_revisar = ['Perovskite_composition_a_ions', 'Perovskite_composition_b_ions', 'Perovskite_composition_c_ions']
    new_Data = Data[~Data[var_names_para_revisar].map(lambda x: '|' in str(x)).any(axis=1)]

    return new_Data

def Remove_nonsingle_layer(Data):
    # only those with the ABX structure.
    Data.loc[:, 'Perovskite_composition_perovskite_ABC3_structure'] = (Data['Perovskite_composition_perovskite_ABC3_structure'].astype(str).str.upper() == 'TRUE')
    Data = Data[Data['Perovskite_composition_perovskite_ABC3_structure'] == True]
    # Eliminar filas con "|" en la columna 'Perovskite_composition_long_form'
    nonsingle_indices = []
    for index, fila in Data.iterrows():
        if "|" in str(fila['Perovskite_composition_long_form']):
            nonsingle_indices.append(index)
    Data = Data.drop(nonsingle_indices)
    #Data = Data.reset_index(drop=True)
    return Data

# -------------------------
def codifica_ABX(Data):
    # Obtains the vectors representing the different perovskite materials appearing in the dataset.
    import pandas as pd

    Data['Perovskite_composition_a_ions'] = Data['Perovskite_composition_a_ions'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_a_ions_coefficients'] = Data['Perovskite_composition_a_ions_coefficients'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_a_ions'] = Data['Perovskite_composition_a_ions'].astype(str)
    Data['Perovskite_composition_a_ions_coefficients'] = Data['Perovskite_composition_a_ions_coefficients'].astype(str)
    mol_A = associate_elements_with_numbers(Data['Perovskite_composition_a_ions'], Data['Perovskite_composition_a_ions_coefficients'])
    mol_A = process_dict_list_A(mol_A)

    Data['Perovskite_composition_b_ions'] = Data['Perovskite_composition_b_ions'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_b_ions_coefficients'] = Data['Perovskite_composition_b_ions_coefficients'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_b_ions'] = Data['Perovskite_composition_b_ions'].astype(str)
    Data['Perovskite_composition_b_ions_coefficients'] = Data['Perovskite_composition_b_ions_coefficients'].astype(str)
    Data['Perovskite_composition_b_ions'], Data['Perovskite_composition_b_ions_coefficients'] = clean_malformed_series(Data['Perovskite_composition_b_ions'], Data['Perovskite_composition_b_ions_coefficients'])
    mol_B = associate_elements_with_numbers(Data['Perovskite_composition_b_ions'], Data['Perovskite_composition_b_ions_coefficients'])
    mol_B = process_dict_list_B(mol_B)

    Data['Perovskite_composition_c_ions'] = Data['Perovskite_composition_c_ions'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_c_ions_coefficients'] = Data['Perovskite_composition_c_ions_coefficients'].apply(lambda s: '' if isinstance(s, str) and ('nan' in s or 'x' in s) else s)
    Data['Perovskite_composition_c_ions'] = Data['Perovskite_composition_c_ions'].astype(str)
    Data['Perovskite_composition_c_ions_coefficients'] = Data['Perovskite_composition_c_ions_coefficients'].astype(str)
    Data['Perovskite_composition_c_ions'], Data['Perovskite_composition_c_ions_coefficients'] = clean_malformed_series(Data['Perovskite_composition_c_ions'], Data['Perovskite_composition_c_ions_coefficients'])
    mol_X = associate_elements_with_numbers(Data['Perovskite_composition_c_ions'], Data['Perovskite_composition_c_ions_coefficients'])
    mol_X = process_dict_list_X(mol_X)

    mol_A.loc[mol_A.notna().any(axis=1)] = mol_A.loc[mol_A.notna().any(axis=1)].fillna(0)
    mol_B.loc[mol_B.notna().any(axis=1)] = mol_B.loc[mol_B.notna().any(axis=1)].fillna(0)
    mol_X.loc[mol_X.notna().any(axis=1)] = mol_X.loc[mol_X.notna().any(axis=1)].fillna(0)
    mol_prvskt = pd.concat([mol_A, mol_B, mol_X], axis=1)

    return mol_prvskt

def process_dict_list_A(dict_list):
    import numpy as np
    import pandas as pd
    result = []
    for entry in dict_list:
        if entry is None or entry == 'nan':
            result.append({'MA': np.nan, 'FA': np.nan, 'Cs': np.nan, 'others_A': np.nan})
            continue

        row = {k: np.nan for k in ['MA', 'FA', 'Cs', 'others_A']}
        other_sum = 0.0

        for key, value in entry.items():
            if pd.isna(value):
                continue
            if key in ['MA', 'FA', 'Cs']:
                row[key] = value
            else:
                other_sum += value

        # only assign others_A if there were any "other" elements
        if other_sum > 0:
            row['others_A'] = other_sum

        result.append(row)

    return pd.DataFrame(result)

def process_dict_list_B(dict_list):
    import numpy as np
    import pandas as pd
    result = []
    for entry in dict_list:
        if entry is None or entry == 'nan':
            result.append({'Pb': np.nan, 'Sn': np.nan, 'others_B': np.nan})
            continue

        row = {k: np.nan for k in ['Pb', 'Sn', 'others_B']}
        other_sum = 0.0

        for key, value in entry.items():
            if pd.isna(value):
                continue
            if key in ['Pb', 'Sn']:
                row[key] = value
            else:
                other_sum += value

        # only assign others_A if there were any "other" elements
        if other_sum > 0:
            row['others_B'] = other_sum

        result.append(row)

    return pd.DataFrame(result)

def process_dict_list_X(dict_list):
    import numpy as np
    import pandas as pd
    result = []
    for entry in dict_list:
        if entry is None or entry == 'nan':
            result.append({'Br': np.nan, 'I': np.nan, 'Cl': np.nan, 'others_X': np.nan})
            continue

        row = {k: np.nan for k in ['Br', 'I', 'Cl', 'others_X']}
        other_sum = 0.0

        for key, value in entry.items():
            if pd.isna(value):
                continue
            if key in ['Br', 'I', 'Cl']:
                row[key] = value
            else:
                other_sum += value

        # only assign others_A if there were any "other" elements
        if other_sum > 0:
            row['others_X'] = other_sum

        result.append(row)

    return pd.DataFrame(result)


def clean_malformed_series(elements_col, numbers_col):
    import pandas as pd

    def is_malformed(s):
        return isinstance(s, str) and (';;' in s or '; ;' in s or s.strip() == '')

    # Create a boolean mask of malformed rows
    mask = elements_col.apply(is_malformed) | numbers_col.apply(is_malformed)
    elements_clean = elements_col.copy()
    numbers_clean = numbers_col.copy()
    elements_clean[mask] = 'nan'
    numbers_clean[mask] = 'nan'

    return elements_clean, numbers_clean

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

# ------
def codifica_DMSO_DMF(Data):
    # de Data, se toma el valor correspondiente a las relaciones de DMSO y DMF, para así retornar el valor de la razón entre ambas.
    import pandas as PD
    import numpy as np

    mis_datos = [[]]
    solventes = Data['Perovskite_deposition_solvents']                # nombre de los solventes
    coeffs = Data['Perovskite_deposition_solvents_mixing_ratios']     # relación de cantidad de esos mismos solventes
    
    for index in range(len(Data['Perovskite_deposition_solvents'])):
        #-fila = str(solventes[index])
        fila = str(solventes.iloc[index])
        #-fila_2 = str(coeffs[index])
        fila_2 = str(coeffs.iloc[index])
        tempo = fila.split('; ')                                   # los nombres de compuestos se separan por comas.
        tempo_2 = fila_2.split('; ')
        if 'Unknown' in tempo or 'nan' in tempo_2 or 'none' in tempo or verifica_symbol_in_solvents(tempo, 'none') or verifica_symbol_in_solvents(tempo, '|'): # busca eliminar filas con errores.
            mis_datos.append([float('nan'), float('nan'), float('nan')])
            continue
        else:
            elementos = split_elements_solvents(tempo)
            cantidades = split_elements_solvents(tempo_2)
            vector = codifica_solventes(elementos, cantidades)
            mis_datos.append(vector)

    Solvents = PD.DataFrame(mis_datos, columns =['DMF', 'DMSO', 'other_solvent'], dtype = float)
    # I want the ratio DMF/DMSO, but many values are DMSO=0; so, I prefer to use log(DMF/DMSO).
    #*Solvents['DMF_DMSO_ratio'] = 0.0           # Initialize column with zeros.
    Solvents['DMF_DMSO_ratio'] = np.nan        #-
    mask = Solvents['DMF'] == 0                #-
    Solvents.loc[mask, 'DMF_DMSO_ratio'] = 0   #- 
    mask = Solvents['DMF'] > 0                 # Mask for safe values (DMF > 0)
    Solvents.loc[mask, 'DMF_DMSO_ratio'] = (np.log10(Solvents.loc[mask, 'DMF']) - np.log10(Solvents.loc[mask, 'DMSO'] + 0.0001)) # Safe computation
    Solvents = Solvents.iloc[1:]
    Solvents = Solvents.reset_index(drop=True)

    return Solvents

def codifica_Tyt(Data):
    import pandas as PD
    import numpy as np

    Temps = Data['Perovskite_deposition_thermal_annealing_temperature']
    tiempos = Data['Perovskite_deposition_thermal_annealing_time']

    Temps = Temps.apply(lambda x: list(map(int, extract_numbers(x))))   # aplica la función extract_numbers(text)
    Temps = [
        [num for num in sublist if num != 0]  # Recorrer cada número en la sublista y eliminar los ceros
        for sublist in Temps                  # Recorrer cada sublista en Temps
    ]

    tiempos = tiempos.apply(lambda x: list(map(int, extract_numbers(x))))   # aplica la función extract_numbers(text)
    tiempos = [
        [num for num in sublist if num != 0]  # Recorrer cada número en la sublista y eliminar los ceros
        for sublist in tiempos                  # Recorrer cada sublista en Temps
    ]

    # Multiplicar elemento a elemento si el tamaño es correcto
    valores_ET = [
        [num1 * num2 for num1, num2 in zip(sublist1, sublist2)]
        for sublist1, sublist2 in zip(tiempos, Temps)
    ]
    
    #-ET = [sum(sublist) for sublist in valores_ET]   # calcula el grado de exposición térmica.
    ET = [sum(sublist) if sublist else np.nan for sublist in valores_ET]

    ET = [ ((float(x))+0.0001) for x in ET]
    ET = np.log10(ET)

    first_elements = [sublist[0] if sublist else np.nan for sublist in Temps]
    df1 = PD.DataFrame(first_elements, columns=['first_Prvskt_annealing_temperature'])
    #first_elements = [sublist[0] for sublist in tiempos]
    first_elements = [sublist[0] if sublist else np.nan for sublist in tiempos]
    df2 = PD.DataFrame(first_elements, columns=['first_Prvskt_thermal_annealing_time'])

    ET = PD.DataFrame(ET, columns=['Perovskite_annealing_thermal_exposure'])
    df = PD.concat([df1, df2, ET], axis=1, join="inner")
     
    return(df)

def extract_numbers(text):
    import re
    return re.findall(r'\d+', text)  # Usa re.findall para obtener todos los números de la cadena


# ---------------------------
# ************ codifica los elementos de la capa hacia la forma de porcentajes. Lo demás queda igual.
def codifica_abx(db):
    # ingresa dataframe con las siguientes columnas:
    # MA; FA; Cs; other_A; Pb; Sn; other_B; Br; I; Cl; other_X;
    # retorna dataframe con los iones A, B y X.
    import pandas as Pandas
    import numpy as NP

    df = Pandas.DataFrame([], columns=['A_ions','B_ions','X_ions','MA_ratio','FA_ratio','Cs_ratio','Pb_ratio','Sn_ratio', 'Br_ratio','I_ratio', 'Cl_ratio'])  # crea dataframe vacío.
    # definición de los IONES.
    A = db.MA - db.FA - db.Cs
    B = db.Pb - db.Sn
    X = db.I  + db.Cl - db.Br
    df["A_ions"] = A
    df["B_ions"] = B
    df["X_ions"] = X
    # para obtener las proporciones de cada elemento.
    suma_1 = db["FA"] + db["MA"] + db["Cs"] + db["others_A"]
    suma_2 = db["Pb"] + db["Sn"] + db["others_B"]
    suma_3 = db["Br"] + db["I"] + db["others_X"]
    
    df["FA_ratio"] = db["FA"]/suma_1
    df["MA_ratio"] = db["MA"]/suma_1
    df["Cs_ratio"] = db["Cs"]/suma_1
    df["Pb_ratio"] = db["Pb"]/suma_2
    df["Sn_ratio"] = db["Sn"]/suma_2
    df["Br_ratio"] = db["Br"]/suma_3
    df["I_ratio"]  = db["I"]/suma_3
    df["Cl_ratio"] = db["Cl"]/suma_3

    return df

# ------------------------------------------------------------------------------
def split_elements_solvents(A):
    lista = []
    for item in A:
        x = item.split(" >> ")
        for element in x:
            lista.append(element)
    return lista

def codifica_solventes(elementos, cantidades):
    temporal = [float(i) for i in cantidades]
    cantidades = temporal
    suma = sum(cantidades)
    
    if 'DMF' in elementos:
        index_DMF = elementos.index('DMF')
        cant_DMF = float(cantidades[index_DMF])/suma
        cantidades.pop(index_DMF)
        elementos.pop(index_DMF)
    else:
        cant_DMF = 0
            
    if 'DMSO' in elementos:
        index_DMSO = elementos.index('DMSO')
        cant_DMSO = float(cantidades[index_DMSO])/suma
        cantidades.pop(index_DMSO)
        elementos.pop(index_DMSO)
    else:
        cant_DMSO = 0
    
    cant_other = sum(cantidades)
    cant_other = cant_other/suma
    
    return [cant_DMF, cant_DMSO, cant_other]

def verifica_symbol_in_solvents(tempo, substring):
    for string in tempo:
        if substring in string:
            flag = True
            return flag
            break
        else:
            flag = False
    return flag


# --------------------------------------------------------------------
# Definir una función para procesar y separar los valores de temperatura y tiempo


def average_numbers_in_cell(value):
    import pandas as pd
    import numpy as np
    import re
    """
    Extracts float numbers (handling commas as decimal separators) from a single cell value
    and returns their average. Returns NaN if no numbers are found.
    """
    if isinstance(value, str):
        # Regular expression to find potential float numbers with '.' or ','
        numbers_str = re.findall(r'\d+[,.]?\d*', value)
        extracted_numbers = []
        for num_str in numbers_str:
            try:
                # Replace comma with dot for consistent float conversion
                num_str_unified = num_str.replace(',', '.')
                extracted_numbers.append(float(num_str_unified))
            except ValueError:
                pass
        if extracted_numbers:
            return np.mean(extracted_numbers)
        else:
            return np.nan
    else:
        return np.nan


def from_csv2dict(file_name_embeddings):
    import csv
    import numpy as np

    embeddings_dict = {}
    # Load from CSV
    with open(file_name_embeddings, mode="r") as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header

        for row in reader:
            key = row[0]
            embedding = np.array([float(x) for x in row[1:]])
            embeddings_dict[key] = embedding

    return embeddings_dict

def Composites_with_LLEs(Data, embed_prvkt_dict):
    import pandas as pd
    import numpy as np

    embed_data = Data['Perovskite_composition_long_form'].map(embed_prvkt_dict)
    # Create a list to store the embedding rows, handling potential NaNs
    embedding_list = []
    for embedding in embed_data:
        if np.isnan(embedding).any():
            embedding_list.append([np.nan] * 4)  # Use NaN for missing or incorrect embeddings
        else:
            embedding_list.append(list(embedding))  # Ensure it's a list

    embed_df = pd.DataFrame(embedding_list, columns=['LLE_1', 'LLE_2', 'LLE_3', 'LLE_4'])
    print('...........')
    print(len(embed_df))
    #-embed_data_jacobsson = pd.concat([Data['Perovskite_composition_long_form'], embed_df], axis=1)
    embed_data_jacobsson = pd.concat([
        Data['Perovskite_composition_long_form'].reset_index(drop=True),
        embed_df.reset_index(drop=True)
        ], axis=1)


    return embed_data_jacobsson


# -------------------------

# def Codifica_A(Data):  
#     # Codifica la parte A de la representación AB-X. # Estructura de salida:  mis_datos = [['MA', 'FA', 'Cs', 'thA']]
#     # los compuestos más comunes son MA, FA y Cs. De deja una columna adicional para otros.
#     import pandas as PD

#     mis_datos = [[]]
#     index = 1
#     elemen_A = Data['Perovskite_composition_a_ions']               # nombre de los compuestos 
#     coeffs_A = Data['Perovskite_composition_a_ions_coefficients']  # medida de cantidad de esos mismos compuestos
#     for index in range(len(Data['Perovskite_composition_a_ions'])):
#         print('-----------')
#         print(index)
#         print(elemen_A[index])
#         print(coeffs_A[index])
#         fila = str(elemen_A[index])
#         fila_2 = str(coeffs_A[index])
#         tempo = fila.split('; ')                                   # los nombres de compuestos se separan por comas.
#         tempo_2 = fila_2.split('; ')
#         if len(tempo)>len(tempo_2) or fila=='nan':                 # verifica q' cada compuesto tenga cantidad. Si-no, se llena con nan.
#             mis_datos.append([float('nan'), float('nan'), float('nan'), float('nan')])
#             continue
#         else:
#             linea = [tempo, tempo_2]
            
#         count_MA = linea[0].count('MA')
#         count_FA = linea[0].count('FA')
#         count_Cs = linea[0].count('Cs')
#         patron = [0, 0, 0, 0]
#         index_delete = []
        
#         if count_MA>0:
#             index_MA = linea[0].index('MA')
#             if linea[1][index_MA]=='x':
#                 patron[0]=float('nan')
#             else:
#                 patron[0] = float(linea[1][index_MA])
#             index_delete.append(index_MA)
            
#         if count_FA>0:
#             index_FA = linea[0].index('FA')
#             if linea[1][index_FA]=='x':
#                 patron[1]=float('nan')
#             else:
#                 patron[1] = float(linea[1][index_FA])
#             index_delete.append(index_FA)

#         if count_Cs>0:
#             index_Cs = linea[0].index('Cs')
#             if linea[1][index_Cs]=='x':
#                 patron[2]=float('nan')
#             else:
#                 patron[2] = float(linea[1][index_Cs])
#             index_delete.append(index_Cs)
 
#         index_delete.sort(reverse=True)   # indices ordenados de forma descendente.
#         for jj in range(0, len(index_delete)):
#             linea[0].pop(index_delete[jj])   # borra desde los últimos hacia los primeros indices para no afectar indices.
#             linea[1].pop(index_delete[jj])
            
#         if len(linea[0])>0:
#             if 'x' in linea[1]:
#                 patron[3]=float('nan')
#             else:
#                 suma = 0
#                 for ii in range(0, len(linea[1])):
#                     suma = suma + float(linea[1][ii])
#                 patron[3] = suma
#         mis_datos.append(patron)

#     df_A = PD.DataFrame(mis_datos, columns =['MA', 'FA', 'Cs', 'others_A'], dtype = float)
#     df_A = df_A.iloc[1:]
#     df_A = df_A.reset_index(drop=True)
#     return df_A

# def Codifica_B(Data):
#     # Estructura de salida. mis_datos = [['Pb', 'Sn', 'other']]
#     import pandas as PD

#     mis_datos = [[]]
#     index = 1
#     elemen_B = Data['Perovskite_composition_b_ions']               # nombre de los compuestos 
#     coeffs_B = Data['Perovskite_composition_b_ions_coefficients']  # medida de cantidad de esos mismos compuestos
#     for index in range(len(Data['Perovskite_composition_b_ions'])):
#         fila = str(elemen_B[index])
#         fila_2 = str(coeffs_B[index])
#         tempo = fila.split('; ')                                   # los nombres de compuestos se separan por comas.
#         tempo_2 = fila_2.split('; ')
#         if len(tempo)>len(tempo_2) or fila=='nan':                 # verifica q' cada compuesto tenga cantidad. Si-no, se elimina.
#             mis_datos.append([float('nan'), float('nan'), float('nan')])
#             continue
#         else:
#             linea = [tempo, tempo_2]
            
#         count_Pb = linea[0].count('Pb')
#         count_Sn = linea[0].count('Sn')
#         patron = [0, 0, 0]
#         index_delete = []
        
#         if count_Pb>0:
#             index_Pb = linea[0].index('Pb')
#             if linea[1][index_Pb]=='x':
#                 patron[0]=float('nan')
#             else:
#                 patron[0] = float(linea[1][index_Pb])
#             index_delete.append(index_Pb)
            
#         if count_Sn>0:
#             index_Sn = linea[0].index('Sn')
#             if linea[1][index_Sn]=='x':
#                 patron[1]=float('nan')
#             else:
#                 patron[1] = float(linea[1][index_Sn])
#             index_delete.append(index_Sn)

#         index_delete.sort(reverse=True)   # indices ordenados de forma descendente.
#         for jj in range(0, len(index_delete)):
#             linea[0].pop(index_delete[jj])   # borra desde los últimos hacia los primeros indices para no afectar indices.
#             linea[1].pop(index_delete[jj])
            
#         if len(linea[0])>0:
#             if 'x' in linea[1]:
#                 patron[2]=float('nan')
#             else:
#                 suma = 0
#                 for ii in range(0, len(linea[1])):
#                     suma = suma + float(linea[1][ii])
#                 patron[2] = suma
#         mis_datos.append(patron)
    
#     df_B = PD.DataFrame(mis_datos, columns =['Pb', 'Sn', 'others_B'], dtype = float)
#     df_B = df_B.iloc[1:]
#     df_B = df_B.reset_index(drop=True)
#     return df_B

# def Codifica_X(Data):
#     # Estructura de salida: mis_datos = [['ID_Index', 'Br', 'I', 'Cl', 'other']]
#     import pandas as PD

#     mis_datos = [[]]
#     index = 1
    
#     elemen_X = Data['Perovskite_composition_c_ions']               # nombre de los compuestos 
#     coeffs_X = Data['Perovskite_composition_c_ions_coefficients']  # medida de cantidad de esos mismos compuestos
#     for index in range(len(Data['Perovskite_composition_c_ions_coefficients'])):
#         fila = str(elemen_X[index])
#         fila_2 = str(coeffs_X[index])
#         tempo = fila.split('; ')                                   # los nombres de compuestos se separan por comas.
#         tempo_2 = fila_2.split('; ')
#         if len(tempo)>len(tempo_2) or fila=='nan':                 # verifica q' cada compuesto tenga cantidad. Si-no, se elimina.
#             mis_datos.append([float('nan'), float('nan'), float('nan')])
#             continue
#         else:
#             linea = [tempo, tempo_2]
            
#         count_Br = linea[0].count('Br')   #<--  Pb**Sn
#         count_I  = linea[0].count('I')    #<--  Sn**I
#         count_Cl = linea[0].count('Cl')   #<--  Cl
#         patron = [0, 0, 0, 0]                #<--
#         index_delete = []
        
#         if count_Br>0:
#             index_Br = linea[0].index('Br')
#             if linea[1][index_Br]=='x':
#                 patron[0]=float('nan')
#             else:
#                 patron[0] = float(linea[1][index_Br])
#             index_delete.append(index_Br)
            
#         if count_I>0:
#             index_I = linea[0].index('I')
#             if linea[1][index_I]=='x':
#                 patron[1]=float('nan')
#             else:
#                 patron[1] = float(linea[1][index_I])
#             index_delete.append(index_I)
            
#         if count_Cl>0:
#             index_Cl = linea[0].index('Cl')
#             if linea[1][index_Cl]=='x':
#                 patron[2]=float('nan')
#             else:
#                 patron[2] = float(linea[1][index_Cl])
#             index_delete.append(index_Cl)

#         index_delete.sort(reverse=True)   # indices ordenados de forma descendente.
#         for jj in range(0, len(index_delete)):
#             linea[0].pop(index_delete[jj])   # borra desde los últimos hacia los primeros indices para no afectar indices.
#             linea[1].pop(index_delete[jj])
            
#         if len(linea[0])>0:
#             if 'x' in linea[1]:
#                 patron[3]=float('nan')
#             else:
#                 suma = 0
#                 for ii in range(0, len(linea[1])):
#                     suma = suma + float(linea[1][ii])
#                 patron[3] = suma

#         mis_datos.append(patron)

#     df_X = PD.DataFrame(mis_datos, columns =['Br', 'I', 'Cl', 'others_X'], dtype = float)
#     df_X = df_X.iloc[1:]
#     df_X = df_X.reset_index(drop=True)
#     return df_X