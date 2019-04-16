noise_type = ["factory"];
snr = [0];

win_len = 320;
win_shift = 160;
fs = 16e3;
useFixedScaFac = 0;
sca_fac = 0;

validation_percentage = 2;
testing_percentage = 1;
file_per_batch = 2;
total_train_steps = 2;
validation_step = 1;
initial_learn_rate = 1e-3;
%learn_rate_decay_fac = 0.95;
useInputNormalization = 1;

mini_batch_size = 1024;
every_train_step = 2;
checkpoint_save_steps = 2;

checkpoint_path_net = '';
checkpoint_path_optimizer = '';

save_training_data = 0;
load_training_data = '';

% gpuDevice;    % the command might throw some error if the graphic driver is outdated
% isGPU = gpuDeviceCount;

isGPU = 0;      % set to 1 to use GPU 

hidden_layer_struct = [512, 512, 512];

speech_path = '.\\data\\clean_speech';
noise_path = '.\\data\\noise';
save_path = '.\\data\\demo';

read_from_list = 0;
train_list_path = '.\\list\\training_list731.txt';
test_list_path = '.\\list\\testing_list87.txt';

adjacent_frame = 1;

