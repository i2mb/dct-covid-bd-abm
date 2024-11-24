# dct-covid-bd-abm
This repository contains the evaluation procedure for result generation and analysis of  the  "Impact of Digital 
Contact Tracing on SARS-CoV-2 Pandemic Control Analysed with Behaviour-driven Agent-based Modelling" paper.

# Installation
This analysis uses the I2MB simulator engine. Please follow these instructions to get all the required packages and 
set up the work environment. The code base is pure Python, so translating this instruction to your preferred Python 
environment should be straightforward. 

## Software requirements
* Python 3.10
* virtualenv
* gunzip

Please install Python 3.10 and virtualenv on your operating system. 
  
## Setup environment
The following instructions create the work environment.

```bash
# Get to a folder where you would like to set up the I2MB environment then follow the instruction.
mkdir i2mb
cd i2mb 

# Get all i2mb repositories
git clone https://github.com/i2mb/dct-covid-bd-abm.git
git clone --branch v0.1.0  https://github.com/i2mb/i2mb-core.git
git clone --branch v0.1.0 https://github.com/i2mb/i2mb-dashboard.git

# We set dct-covid-bd-abm as our main working directory
cd dct-covid-bd-abm

# Create output directories
mkdir -p ./data/simulations ./figures   ./data/contact_validation

# Setup and activate virtual environment
virtualenv .venv

# Install dependencies
pip install -r requirements.txt
pip install -r ../i2mb-core/requirements.txt
pip install -r ../i2mb-dashboard/requirements.txt

```

# Activate the environment
The following steps are required in order to run the code when a new session is created.

```bash
source .venv/bin/activate
```

Set `PYTHONPATH` environment variable. In the future, we might do this with python packages. But for transparency, 
and provide you with complete access to the code, we suggest using `PYTHONPATH` to link all the I2MB repositories.
```bash
export PYTHONPATH=../i2mb-core:../i2mb-dashboard:${PYTHONPATH}

# Test installation, the following commands should return no error.
python -c "import i2mb"
./contact_tracing_eval.py --help
```
Read the help output from `./contact_tracing_eval.py`. It explains what each argument does. The following arguments are 
dependent on the `--configs` parameter: `-c`, `-S`, `-e`, so the help message will reflect the items defined in the 
selected configuration file.

# Generate data
The following instructions will run the configurations necessary to perform the analysis.

Please notice that run time will depend on resource availability. We used three servers with 40 cores and  64GB of 
ram, with that configuration it took us about two weeks of continuous running to generate all the material. After 
the run and post-processing were finished, about 500GB of data were generated.

```bash

# Compute baselines
python contact_tracing_eval.py -j 40 -d data/simulation --configs=dct_covid_bd_abm/configs/evaluation_config.py \ 
      -c stage_1 --num-runs 50 --num-agents 1000

# DCT parameter search
python contact_tracing_eval.py -j 40 -d data/simulation --configs=dct_covid_bd_abm/configs/evaluation_config.py \ 
      -c stage_1_grid --num-runs 50 --num-agents 1000

# NPIs realistic behaviour
python contact_tracing_eval.py -j 40 -d data/simulation --configs=dct_covid_bd_abm/configs/evaluation_config.py \ 
      -c stage_2_best_main_parameters --num-runs 50 --num-agents 1000

# NPIs optimal behaviour
python contact_tracing_eval.py -j 40 -d data/simulation --configs=dct_covid_bd_abm/configs/evaluation_config.py \
      -c stage_2_perfect_behaviour --num-runs 50 --num-agents 1000

# Contact tracing validation
python contact_tracing_eval.py -j 40 -d data/simulation \
      --configs=dct_covid_bd_abm/configs/contact_validation_config.py \
      -c  stage_2_perfect_behaviour --num-runs 50 --num-agents 1000
      
#  Convert csv files to feather file format for faster processing
./csv_to_parquete_converter.py  data/simulation
```
# Generate figures and tables from the paper.
To create the figures and tables form teh paper additional datasets are required. The following instructions will 
provide access to the datasets from [Extrasensory](https://doi.org/10.1109/MPRV.2017.3971131).

```bash
mkdir -p dct_covid_bd_abm/simulator/assets/activities/extrasensory
curl --ouput dct_covid_bd_abm/simulator/assets/activities/ExtraSensory.per_uuid_features_labels.zip \
     http://extrasensory.ucsd.edu/data/primary_data_files/ExtraSensory.per_uuid_features_labels.zip

# extract the files  
unzip ExtraSensory.per_uuid_features_labels.zip -d dct_covid_bd_abm/simulator/assets/activities/extrasensory
EXTRASENSORY=dct_covid_bd_abm/simulator/assets/activities/extrasensory/
for csv_file in $EXTRASENSORY/*.gz; do  gunzip   $csv_file; done

# Clear up some space
rm dct_covid_bd_abm/simulator/assets/activities/ExtraSensory.per_uuid_features_labels.zip

#  Convert to feather file format for faster processing
./csv_to_parquete_converter.py $EXTRASENSORY
```

to generate all figures and tables run the following command:
```bash
./generate_figures.py  
```

To generate a specific figure or table, please edit `./generate_figures.py` and add only the desired plot or table.
`./generate_figures.py` always prints a lists of all available tables or figures.

THe following example displays figures 2 and 3.

```python
if __name__ == "__main__":
    list_available_items()
    main(figures_names=[
        "2_grid_search_panel",
        "3_parallel_areas"])
```
