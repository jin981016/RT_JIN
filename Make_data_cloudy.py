from RT import *

Lumin = np.array([42.0, 42.5,43.0])
metals = 0.1
Column_density_order = 20.5
atom = np.array(['CIV','HeII'])

save_path = f'/home/jinlim/update_RT/CLOUDY_data/'

for atom_i in atom:
    for Lumin_i in Lumin:
        path = resolve_column_density_path(Lumin_i, metals, Column_density_order, 'CIV/CLOUDY_QSO')
        make_data_file(save_path, path, atom_i, Lumin_i, metals, Column_density_order)