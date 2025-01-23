from yade import pack, utils, plot, export
import pickle
import numpy as np

# -----------------------------------------------------------------------------#
# Functions called once
# -----------------------------------------------------------------------------#

def create_materials():
    '''
    Create materials.
    '''
    O.materials.append(FrictMat(young=E, poisson=Poisson, frictionAngle=atan(0.5), density=density, label='frictMat'))
    O.materials.append(FrictMat(young=E, poisson=Poisson, frictionAngle=0, density=density, label='frictlessMat'))

# -----------------------------------------------------------------------------#

def create_grains():
    '''
    Recreate level set from data extrapolated with phase field output.
    '''
    print("Creating level set")

    for i_data in range(1, n_data_base+1):
        # load data
        with open('data/level_set_part'+str(i_data)+'.data', 'rb') as handle:
            dict_save = pickle.load(handle)
        L_sdf_i_map = dict_save['L_sdf_i_map']
        L_x_L = dict_save['L_x_L']
        L_y_L = dict_save['L_y_L']
        L_rbm = dict_save['L_rbm']
        extrude_z = dict_save['extrude_z']

        # create grain
        for i_grain in range(len(L_sdf_i_map)):
            # grid
            spacing = L_x_L[i_grain][1]-L_x_L[i_grain][0]
            grid = RegularGrid(
                min=(min(L_x_L[i_grain]), min(L_y_L[i_grain]), -extrude_z*spacing/2),
                nGP=(len(L_x_L[i_grain]), len(L_y_L[i_grain]), extrude_z),
                spacing=spacing
            )  
            # grains
            O.bodies.append(
                levelSetBody(grid=grid,
                            distField=L_sdf_i_map[i_grain].tolist(),
                            material=0)
            )
            O.bodies[-1].state.blockedDOF = 'zXYZ'
            O.bodies[-1].state.pos = L_rbm[i_grain]
            O.bodies[-1].state.refPos = L_rbm[i_grain]
        
# -----------------------------------------------------------------------------#

def compute_dt():
    '''
    Compute the time step used in the DEM step.
    '''
    O.dt = 0.2*SpherePWaveTimeStep(radius=radius, density=density, young=E)

# -----------------------------------------------------------------------------#

def create_engines():
    '''
    Create engines.

    Overlap based on the distance

    Ip2:
        kn = given
        ks = given    

    Law2:
        Fn = kn.un
        Fs = ks.us
    '''
    O.engines = [
            VTKRecorder(recorders=["lsBodies"], fileName='./2_', iterPeriod=0, multiblockLS=True, label='initial_export'),   
            PyRunner(command='check()',iterPeriod=1, label='checker')
    ]

# -----------------------------------------------------------------------------#
# Functions called multiple times
# -----------------------------------------------------------------------------#

def check():
    '''
    Try to detect a steady-state.
    A maximum number of iteration is used.
    '''
    if O.iter > 1:
        O.pause() # stop DEM simulation
        
# -----------------------------------------------------------------------------#
# Load data
# -----------------------------------------------------------------------------#

# from main
radius = 50
E = 7e13
Poisson = 0.3
kn = E*radius
ks = E*Poisson*radius
force_applied = 0.05*E*radius**2
density = 2000 

# main information
with open('data/level_set_part0.data', 'rb') as handle:
    dict_save = pickle.load(handle)
L_x = dict_save['L_x']
L_y = dict_save['L_y']
L_pos_w = dict_save['L_pos_w']
m_size = dict_save['m_size']
n_data_base = dict_save['n_data_base']

# -----------------------------------------------------------------------------#
# Plan simulation
# -----------------------------------------------------------------------------#

# materials
create_materials()
# create grains and walls
create_grains()
# Engines
create_engines()
# time step
compute_dt()

# -----------------------------------------------------------------------------#
# MAIN DEM
# -----------------------------------------------------------------------------#

O.run()
O.wait()

# -----------------------------------------------------------------------------#
# Output
# -----------------------------------------------------------------------------#

L_surfNodes = []
for b in O.bodies:
    if isinstance(b.shape, LevelSet):
        L_surfNodes.append(len(b.shape.surfNodes))

# transmit data
dict_save = {
    'L_surfNodes': L_surfNodes
}
with open('data/output_level_set.data', 'wb') as handle:
    pickle.dump(dict_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
