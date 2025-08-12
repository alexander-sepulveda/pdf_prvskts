"""
We use the dataset created by Jasper, Unger. et. al. in order to extract those descriptors related
to perovskite solar cell structure and systesis process.
Descriptors :
    - 'Cs', 'MA', 'FA', 'thA': proportion of Cs, MA, FA & other elements (Ion-A).
    - 'Pb', 'Sn', 'th_B' : proportion of Pb, Sn & other elements (Ion-B).
    - 'Br', 'I', 'Cl', 'th_X' : proportion of Br, I, Cl and other elements (Ion-X).
    - 'A': number of ions in A
    - 'B': number of ions in A B
    - 'X': number of ions in A X
    - LLEs
    - Thickness
    - BandGap
    - 'DMSO', 'DMF', 'th_solvent' : proportion of DMSO, DMF y other solvents.
    - DMSO_DMF_ratio : ratio DMSO/DMF
    - 'Temp_i' : annealing temperatures
    - 't_i' : annealing times
    - 'G-T' : thermal budget
we are going to work with single layer perovskites.
"""

def main():
    # 
    import pandas as PD
    import sys
    import os
    from pathlib import Path
    sys.path.insert(0, './src')  # add src to Python path
    import ABX

    # --- input parameters
    base_dir = Path(__file__).parent.parent    # project location
    filename_descriptors = base_dir / "data" / "Descriptores_prvskts.csv"
    file_name_data = base_dir / "data" / "Perovskite_database_content_all_data.csv"
    filename_results = base_dir / "data" / "Descriptors_Perovskite_Database.xlsx"
    file_name_embeddings = base_dir / "data" / "embeddings_cosine.csv"
    # --------------------

    # read the variables names.
    dtype_spec = {'Variable': str, 'Description': str}
    df = PD.read_csv(filename_descriptors, sep=';', dtype = dtype_spec)  # Replace with the actual filename
    descriptors_names = df['Variable'].tolist()
    # read the original prvskt data and take out the variabls
    data = PD.read_csv(file_name_data, low_memory=False)
    Data = data[descriptors_names]
    Data = ABX.Remove_nonsingle_layer(Data)   # only single layer prvskts

    df_ABX = ABX.codifica_ABX(Data)           # take out solvens.
    Solvents = ABX.codifica_DMSO_DMF(Data)
    TtE = ABX.codifica_Tyt(Data)              # temperatures, times y thermal budget.
    hola = ['Perovskite_band_gap', 'Perovskite_thickness', 'Cell_area_measured', 'Ref_ID', 'Cell_stack_sequence']
    df_otros_prvskts = Data[hola]
        
    salidas = ['JV_default_Voc', 'JV_default_Jsc', 'JV_default_FF', 'JV_default_PCE', 'Stability_PCE_T80']
    df_salidas = Data[salidas]
    
    hola = ['ETL_deposition_synthesis_atmosphere_relative_humidity', 'HTL_deposition_synthesis_atmosphere_relative_humidity', 
    'Perovskite_deposition_thermal_annealing_relative_humidity', 'Perovskite_deposition_synthesis_atmosphere_relative_humidity',
    'HTL_thickness_list', 'ETL_thickness', 'Backcontact_thickness_list']
    df_humidity =  Data[hola]
    df_humidityAA = df_humidity.map(ABX.average_numbers_in_cell)

    # the perovskite material is represented by a 4-dimensional vector whose  transformation was obtained Local Linear Embedding.
    # In this casit using a dictionary to apply the tranformation; but, the method "xx-xx" can also be used to
    # carry out that transformation.
    embed_prvkt_dict = ABX.from_csv2dict(file_name_embeddings)
    embed_data = ABX.Composites_with_LLEs(Data, embed_prvkt_dict)
    
    # join all variables.
    hola = ['ETL_stack_sequence', 'ETL_deposition_procedure', 'HTL_stack_sequence', 'HTL_deposition_procedure', 'Perovskite_composition_perovskite_ABC3_structure', 'Backcontact_stack_sequence',  'Perovskite_deposition_procedure']
    hola_hola = Data[hola]
    dfs = [df.reset_index(drop=True) for df in [embed_data, df_ABX, Solvents, TtE, df_otros_prvskts, df_humidityAA, hola_hola, df_salidas]]
    df_total = PD.concat(dfs, axis=1)
    
    df_total.to_excel(filename_results, index=False)
    print("Results saved in:", filename_results)
    print("Number of observations:", len(df_total))


if __name__ == "__main__":  # only run main() if the file is run directly.
    main()