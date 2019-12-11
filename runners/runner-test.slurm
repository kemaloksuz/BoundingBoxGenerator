#!/bin/bash
#SBATCH -p akya-cuda
#SBATCH -A bcam
#SBATCH -J setup
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --workdir=/truba_scratch/bcam/output
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bariscancam@gmail.com
#SBATCH --time=15-00:00:00
#SBATCH --output=/truba_scratch/bcam/output/slurm-%j.out
#SBATCH --error=/truba_scratch/bcam/output/slurm-%j.err

###################  Init. CUDA ##########################
export CUDA_HOME=/truba_scratch/bcam/software/cuda-10.0
export PATH=/truba_scratch/bcam/software/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/bcam/software/cuda-10.0/lib64
######################################################################

###################  Init. conda ##########################
_conda_setup="$('/truba_scratch/bcam/software/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/truba_scratch/bcam/software/miniconda3/etc/profile.d/conda.sh" ]; then
	. "/truba_scratch/bcam/software/miniconda3/etc/profile.d/conda.sh"
    else
	export PATH="/truba_scratch/bcam/software/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
export PATH="/truba_scratch/bcam/software/miniconda3/condabin:$PATH"
######################################################################

###################  Set open-mmlab env ##########################
source activate open-mmlab
######################################################################

###################  Run ##########################
cd /truba/home/bcam/imgworkspace/mmdetection
python setup.py develop
######################################################################

