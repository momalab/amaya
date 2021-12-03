# Amaya

## Overview
Amaya is an external non-intrusive malware detection tool that uses the Joint Test Action Group (JTAG) port for collecting text sections of processes directly from the main memory and ML to detect malware.

## Requirements
Linux<br />
Lauterbach TRACE32<br />
Required python libraries

## Instructions
#### Configuring Lauterbach helper library
  * In libraries/lauterbach-config:
    - Change **T32_PATH** to point to t32marm-qt binary
    - Change **T32_CONFIG_PATH** to the configuration file inside the Lauterbach directory (config.t32)
  * In libraries/bbb-linux-aware:
    - Change kernel binary path (Line 18: Data.LOAD.Elf)
    - Change TASK.CONFIG path (Line 26)
    - Change MENU.ReProgram path (Line 27)
#### Feature extraction from binary files
  * In model_training/feature_extraction.py:
    - Change **DATASET_PATH** to the directory with the binary files
  * Features:
    - model_training/entropy_raw_2d/: Stores the entropy values
    - model_training/string_info/: Stores the string histogram
    - model_training/syscall_info/: Stores syscall histogram
#### Feature compression:
  * In model_training/feature_compression.py:
    - Change **datasetMalBasePath** to the directory with malware features
    - Change **datasetGoodBasePath** to the directory with goodware features
    - Change **resultPath** to the directory for result storage
#### Result verification:
  * Run result_calculation.py which gets features from dataset directory and displays the result on terminal.
#### Non-intrusive runtime process memory extraction:
  * Run start_lauterbach.py after configuring Lauterbach helper library
  * Run extract_pmem_nonintrusive.py, which stores extracted memory in extracted_files directory.
  
## Cite us

If you like the work, please cite our DATE' 21 paper:<br /><br />
@inproceedings{rajput2021towards,
  title={Towards Non-intrusive Malware Detection for Industrial Control Systems},
  author={Rajput, Prashant Hari Narayan and Maniatakos, Michail},
  booktitle={2021 Design, Automation \& Test in Europe Conference \& Exhibition (DATE)},
  pages={1703--1706},
  year={2021},
  organization={IEEE}
}

## Contact us

For more information or help with the setup, please contact Prashant Rajput at **prashanthrajput@nyu.edu**
