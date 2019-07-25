# Locations
## Folders
- LinearA_trained contains a pretrained model. 
- FTIR_to_electrolyte_composition contains code
    - models.py defines how the dataset is stored.
    - management/commands/LinearA_run.py is where the code functionality is defined
     
## Files
- .gitattributes can be ignored
- .gitignore can be ignored
- db.sqlite3 contains the dataset
- LICENSE contains the license
- manage.py is the entry point for the program.
- requirements.txt contains the requirements. (excluding python 3.6)

# Source Code
see FTIR_to_electrolyte/management/commands/LinearA_run.py

# Usage

There are many possible ways to call the code:
1.	After having put some measurement files in a directory, 
    the model can be called by specifying this directory. 
    It will return an excel file with the predicted mass ratios,
    as well as a directory with a graph of the reconstructed spectra
    together with the original data for each input file
     (the extension of the files will be changed from .asp to .png). 
     This is the __run_on_directory__ option. 
     For convenience, an excel sheet with the numerical data
      for measured and reconstructed spectra is outputted as well 
      so the user can make their own graphs (e.g. for publication).
2.	The training can be run on the calibration dataset,
 and thus the model can be updated. 
 This is the __train_on_all_data__ option.
3.	Cross-validation studies of the model can be run to
 evaluate it on the calibration dataset. 
 This is the __cross_validation__ option. 
4.	The figures in this paper can be reproduced. This is the 
__paper_figures__ option.
5.	There is a functionality to create a dataset,
 allowing the extension of the dataset,
  but it might require some modification to adapt to a different lab's workflow. 
  This is the __create_dataset__ option. It is not documented because the way to specify 
  the known weight ratios will have to change for new data. 


here are some examples of calling the code:
- "python manage.py LinearA_run --mode=run_on_directory --logdir=LinearA_trained --input_dir=test_input_data --output_dir=Output" 
will run the trained model stored in folder 'LinearA_trained' on all the .asp files found in folder 'test_input_data', putting the output in folder 'Output'

- "python manage.py LinearA_run --mode=train_on_all_data --logdir=LinearA_trained" 
will train a model on the complete dataset and store the trained model in folder 'LinearA_trained'

- "python manage.py LinearA_run --mode=cross_validation --test_ratios=0.3 --cross_validation_dir=Cross_LinearA"
will run a single training procedure with 30 percent of the data in the test set, storing the result in folder "Cross_LinearA".
  
- "python manage.py LinearA_run --mode=paper_figures --cross_validation_dir=Cross_LinearA"
will produce the figures from the paper for the cross validation runs stored in folder "Cross_LinearA".



# Requirements
see requirements.txt
install the requirements by running on the command line "pip install <something>" with <something substituted for a requirement.



# Understanding the database
An in-depth guide to the dataset and how to access it in python is written at the begining of the file
FTIR_to_electrolyte_composition/models.py 

# Understanding the model that maps FTIR spectra to the mass ratios of various components
This README contains valuable information about how to call the program to accomplish various operations (train, run on a directory, do cross-validation, etc..)
For more in-depth documentation of how the model is implemented, see
FTIR_to_electrolyte_composition/management/commands/LinearA_run.py