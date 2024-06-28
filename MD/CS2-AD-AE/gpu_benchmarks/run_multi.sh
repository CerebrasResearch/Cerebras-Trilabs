#!/bin/bash
#SBATCH --account=xxx
#SBATCH --nodes=32
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --core-spec=16
#SBATCH --ntasks-per-node=8
#SBATCH --threads-per-core=1
#SBATCH --job-name=benchmarking
#SBATCH --output=lammps_snap.%j.out

module load craype-accel-amd-gfx90a
module load rocm/5.7.1
module load cray-mpich/8.1.27

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

#rm log.*

# Enable GPU MPI buffers
export MPICH_GPU_SUPPORT_ENABLED=1

# Make sure the appropriate NIC is chosen per MPI rank.
export MPICH_OFI_NIC_POLICY=NUMA

# Announce and then actually run MY_EXE for each MPI rank
echo "Starting actual srun now: `date '+%y%m%d_%H%M:%S'`"

EXE=./lmp_frontier_kokkos
INPUT=in.bench

for elem in Cu Ta W ; do
  if [ "$elem" == "Cu" ]; then
    nx=174
    ny=192
    nz=6
  else
    nx=256
    ny=261
    nz=6
  fi

  for bc in 0 ; do

    /usr/bin/time -f'%C\nTook %e secs' srun -u -N1 -n1 -c1 --cpu-bind=map_cpu:50 --gpus=1 ${EXE} -in ${INPUT} \
      -k on g 1 -sf kk -pk kokkos neigh full newton off \
      -var element $elem \
      -var nx $nx -var ny $ny -var nz $nz -var pbc $bc \
      -log bench.log.$elem.${nx}x${ny}x${nz}.pbc_$bc.gpus_1

    for NNODES in 1 2 4 8 16 32 ; do
      /usr/bin/time -f'%C\nTook %e secs' srun -u -N ${NNODES} --ntasks-per-node=8 --cpus-per-task=6 --gpus-per-task=1 --gpu-bind=closest ${EXE} -in ${INPUT} \
      -k on g 1 -sf kk -pk kokkos neigh full newton off \
      -var element $elem \
      -var nx $nx -var ny $ny -var nz $nz -var pbc $bc \
      -log bench.log.$elem.${nx}x${ny}x${nz}.pbc_$bc.gpus_$((NNODES*8))
    done
  done
done

# All done!
endTS=`date '+%s'`
echo "JobID ${SLURM_JOBID} on ${SLURM_NNODES} nodes ended: `date '+%y%m%d_%H%M:%S'`: $((endTS - startTS)) secs"

