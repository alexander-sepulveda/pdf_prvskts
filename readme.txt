
The overall objective of this work is to model perovskite solar cells by using a probability density function, specifically, we represented the density as a mixture of Gaussians. The process involves several stages. First, the absorber material (perovskite) of the solar cell is represented together with information about the synthesis process of the corresponding device. The material representation is obtained through Local Linear Embeddings (LLE) using the script 'LLE_prvskt_material.py'. Subsequently, the script 'synthesis_descriptors.py' is employed to generate a dataset containing the descriptors for each solar cell device (observation). The dataset consists of descriptors obtained from the Perovskite Database Project. A coding process was applied by which each perovskite material is represented by a four-dimensional vector derived through an LLE method. Additional descriptors capture parameters associated with the synthesis process of the perovskite solar cell.

The GMM model is then trained (using 'pdf_gmm.py') on the newly obtained data and its performance in regression tasks is evaluated against the XGBoost method. The same resulting GMM model is further applied to conditional data generation tasks. In this setting, the material information is assumed to be known, and a target PCE is specified to generate plausible configurations (conditional_generator.py). By leveraging the probability density function, the approach seeks to identify plausible synthesis conditions for new materials.

Additionally, illustrative algorithms are provided to highlight key aspects of this work. The script 'toy_gmm.py' employs two variables (Jsc and Eg) to demonstrate the modeling and regression process using GMMs. The script 'toy_LLE.py' shows the results of representing materials via LLE, including cases not contained in the Perovskite Database Project.

- LLE_prvskt_material.py : Material representation through Local Linear Embedding (LLE).
- synthesis_descriptors.py : It obtains a new dataset consisting of descriptors from the Perovskite Database Project.
- conditional_generator.py : It generates conditional samples; that is, generating new samples from partial information.
- 'toy_gmm.py' depicts the modeling process using GMMs.
- 'toy_LLE.py' shows how materials can be represented by using LLE.

$ python3 scripts/LLE_prvskt_material.py
