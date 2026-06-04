from RT import *

Lumin = np.array([42.0, 42.5])
metals = 0.1
Column_density_order = 20.5
atom = 'CIV'

for Lumin_i in Lumin:
    path = resolve_column_density_path(Lumin_i, metals, Column_density_order, 'CIV/CLOUDY_QSO')
    make_data_file(path, atom, Lumin_i, metals, Column_density_order)