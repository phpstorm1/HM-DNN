% Matlab code to process noisy wav using trained network
% This is the primary code to run for processing noisy wav files for keyword spotting evaluation
% To run the code, specify the noisy_data_path and save_results_path

addpath(genpath('./feat'));
addpath(genpath('./dnn'));
config;

noisy_data_path = 'D:\coding\Matlab\\tmp\\tmp';
save_results_path = 'D:\coding\Matlab\\tmp\\tmp_testing';

% load the net
if isempty(checkpoint_path_net)
	error('checkpoint_path_net is empty');
else
	load(checkpoint_path_net);
end

% load the optimizer
if isempty(checkpoint_path_optimizer)
	error('checkpoint_path_optimizer is empty');
else
	load(checkpoint_path_optimizer);
end

wav_file_list = dir([noisy_data_path, filesep, '**', filesep, '*.wav']);

for i=1:length(wav_file_list)
	cur_wav_path = [wav_file_list(i).folder, filesep, wav_file_list(i).name];
	fprintf('processing file: %s \n', cur_wav_path);
	[noisy_wav, fs] = audioread(cur_wav_path);

	pre_folder = regexp(wav_file_list(i).folder, filesep, 'split');
	pre_folder = char(pre_folder(length(pre_folder)));

	max_amp = max(abs(noisy_wav));
	
	noisy_wav = resample(noisy_wav, 16e3, fs);
    noisy_wav = noisy_wav / max(abs(noisy_wav));
    [testing_feat, testing_label] = get_training_data(...
						noisy_wav, ...
						noisy_wav, ...
						'ARastaplpMfccGf',...
						win_len,...
						win_shift,...
						fs,...
						useFixedScaFac,...
						sca_fac);
    testing_feat = testing_feat';

    if useInputNormalization
        testing_feat_norm = mean_var_norm_testing(testing_feat, ...
        										net.norm_mu, ...
												net.norm_std);
        testing_feat_win = win_buffer(testing_feat_norm, adjacent_frame);
    else
        testing_feat_win = win_buffer(testing_feat, adjacent_frame);
    end

    testing_predict = predict_from_net(net.layers, ...
        testing_feat_win, ...
        optimizer);
    testing_predict = gather(testing_predict');

    testing_estimated = wav_synthesis(testing_predict, ... 
										noisy_wav, ...
										fs, ...
										win_len, ...
										win_shift);

    testing_estimated = testing_estimated * max_amp; 


    cur_save_path = [save_results_path, filesep, pre_folder];
    if(~exist(cur_save_path))
    	mkdir(cur_save_path);
    end

    audiowrite([cur_save_path, filesep, wav_file_list(i).name], testing_estimated, 16e3);

end
