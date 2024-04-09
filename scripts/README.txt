When edditing text files in windows, linejump character  '\r\n' is problematic

If running a .slurm script shows this problem: 

$ sbatch slurm_test.slurm
sbatch: error: Batch script contains DOS line breaks (\r\n)
sbatch: error: instead of expected UNIX line breaks (\n).

Then conver the file using: 

dos2unix slurm_test.slurm

IF dos2unix is not available in the server (my case) the manually remove the \r characters:

tr -d '\r' < slurm_test.slurm > slurm_test_unix.slurm

( 
    tr -d '\r' <#old_windos_file.slurm#> #new_unix_file.slurm#  
) 