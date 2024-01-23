import subprocess
from typing import Text
import pandas as pd
import re
import numpy as np


'''
Bunch all service utils here
directory cleanup
mpi and omp run functions
'''

  

# small utility to clean up backup files:

def clean() -> None:
    print(subprocess.run("rm -f \#*; rm -fr bck.*",shell=True))

# run with mpi/ srun / singularity / gpu
def run_mpi_local(run_cmd, **kwargs):
    return subprocess.run("OMP_NUM_THREADS={} mpiexec -n {} --oversubscribe {}".format(kwargs['omp_threads'], kwargs['mpi_jobs'], run_cmd),shell=True)


def run_mpi_slurm(run_cmd, **kwargs):
    return subprocess.run("srun {} singularity run {} {} mpirun {} {} ".format(
        ' '.join(kwargs['srun_args']),
        ' '.join(kwargs['singularity_args']),
        kwargs['docker_filename'],
        ' '.join(kwargs['mpirun_args']),
        run_cmd),shell=True)
    
                    
def run_omp_local(run_cmd, **kwargs):
    return subprocess.run("set OMP_NUM_THREADS={} && {}".format(kwargs['omp_threads'] , run_cmd), shell=True)

def run_omp_slurm(run_cmd, **kwargs):
    return subprocess.run("singularity run --cleanenv --home $(pwd) --env OMP_NUM_THREADS={} {} {}".format(kwargs['omp_threads'] , kwargs['docker_filename'], run_cmd), shell=True)




def multicolvar_to_pandas(filename: str = 'MULTICOLVAR.xyz', titles: list = []):
    """
    titles: list of column names
    filename: path of multicolvar file
    Reads multicolvar file filename
    returns pandas data frame with column names titles 
    for each colvar
    """
    return pd.DataFrame(multicolvar_to_numpy(filename),columns=titles)

def multicolvar_to_numpy(filename = 'MULTICOLVAR.xyz'):
    """
    
    """
    index = 0
    with open(filename,'r') as f:
        lines = f.readlines()
        number_of_variables= int(lines[0].strip())
        row = []
        for line in lines:
            line = line.strip()
            if re.search('^'+str(number_of_variables),line):
                index+=1
                row=[]
            elif re.search('^X',line):
                row.append(float(line.split(' ')[-1]))
                if len(row)== number_of_variables:
                    if index == 1:
                        multicolvar_np = np.atleast_2d(np.array(row))
                    else:
                        multicolvar_np = np.append(multicolvar_np, np.atleast_2d(np.array(row)), axis=0)
            else:
                continue

    return multicolvar_np


def return_all_idx(list,item):
    """
    Return all indices of item in list or an empty list if non found
    """
    return [i for i,l in enumerate(list) if l==item]


def return_plumed_cv_file(
        reference_file: str = 'reference.pdb',
        descriptors: list = [],
        multicolvar_name: str = 'MULTICOLVAR_F.xyz',
        stride: int = 1,
        hbonds_list: list = [],
        r_0: float = 3.3,
        colvar_name: str = 'COLVAR_F') -> str:
    # returnes plumed text for cv collection 
    return """
    # vim:ft=plumed
    # calculate  CVs for HLDA 
    MOLINFO STRUCTURE={} MOLTYPE=protein
    WHOLEMOLECULES RESIDUES=all MOLTYPE=protein
    rmsd: RMSD REFERENCE={} TYPE=OPTIMAL NOPBC
    e2e: DISTANCE ATOMS=@N-0,@O-9 NOPBC
    rg: GYRATION TYPE=RADIUS ATOMS=@back-0,@back-1,@back-2,@back-3,@back-4,@back-5,@back-6,@back-7,@back-8,@back-9 NOPBC
    # backbone centers
    {}
    
    # sidechain centers 
    {}
    
    # dihedrals
    psi0: TORSION ATOMS=@psi-0 NOPBC
    phi1: TORSION ATOMS=@phi-1 NOPBC
    psi1: TORSION ATOMS=@psi-1 NOPBC
    phi2: TORSION ATOMS=@phi-2 NOPBC
    psi2: TORSION ATOMS=@psi-2 NOPBC
    phi3: TORSION ATOMS=@phi-3 NOPBC
    psi3: TORSION ATOMS=@psi-3 NOPBC
    phi4: TORSION ATOMS=@phi-4 NOPBC
    psi4: TORSION ATOMS=@psi-4 NOPBC
    phi5: TORSION ATOMS=@phi-5 NOPBC
    psi5: TORSION ATOMS=@psi-5 NOPBC
    phi6: TORSION ATOMS=@phi-6 NOPBC
    psi6: TORSION ATOMS=@psi-6 NOPBC
    phi7: TORSION ATOMS=@phi-7 NOPBC
    psi7: TORSION ATOMS=@psi-7 NOPBC
    phi8: TORSION ATOMS=@phi-8 NOPBC
    psi8: TORSION ATOMS=@psi-8 NOPBC
    phi9: TORSION ATOMS=@phi-9 NOPBC

    # use multicolvars to calculate distances 
    D1: DISTANCES GROUP={} NOPBC
    DUMPMULTICOLVAR DATA=D1 FILE={} STRIDE={}
    # hydrogen bonds contactmap 
    cmap: CONTACTMAP {} SWITCH={{RATIONAL R_0={}}}  
    
    PRINT ...
    ARG=rmsd,e2e,rg,psi0,phi1,psi1,phi2,psi2,phi3,psi3,phi4,psi4,phi5,psi5,phi6,psi6,phi7,psi7,phi8,psi8,phi9,cmap.*
    FILE={} STRIDE={}
    ... PRINT
    """.format(reference_file,
               reference_file,
                '\n\t'.join(['b{}: CENTER ATOMS=@back-{} NOPBC'.format(d[1],d[1]) for d in descriptors if d[0] == 'b']),
                '\n\t'.join(['s{}: CENTER ATOMS=@sidechain-{} NOPBC'.format(d[1],d[1]) for d in descriptors if d[0] == 's']),
                ','.join(descriptors),
                multicolvar_name,
                stride,
                ' '.join(['ATOMS{}={},{} '.format(idx+1,i+1,j+1) for idx,(i,j) in enumerate(hbonds_list)]),
                r_0,
                colvar_name,
                stride)


if __name__=='__main__':
    pass    