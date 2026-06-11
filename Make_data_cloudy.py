from RT import *

Lumin = np.array([42.0, 42.5])
metals = 0.1,1.0
Column_density_order = 20.5
atom = np.array(['CIV','HeII'])

save_path = f'/home/jinlim/spatial_RT_run/'

for metal_i in metals:
	for atom_i in atom:
		for Lumin_i in Lumin:
        		path = resolve_column_density_path(Lumin_i, metal_i, Column_density_order, 'CIV/CLOUDY_QSO')
       			make_data_file(save_path, path, atom_i, Lumin_i, metal_i, Column_density_order)
