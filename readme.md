# file structure
    |- data
    |     |- clean_speech   <- speech wav files
    |     |- noise          <- noise wav files
    |- dnn
    |- feat
    |- list                 <- contains lists of speech file that will be used in training
    |- config.m             <- configuration w.r.t. training, network structure, etc.
    |- main.m               <- run this code for training & testing
    |- proc_noisy.m         <- uses pre-trained network to process wav files

# how to run
  1. prepare the training data
  2. tune the parameters in config.m
  3. run main.m
