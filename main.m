addpath(genpath('.\\feat'));
addpath(genpath('.\\dnn'));

% load configurations
config;

% write log file
t = datestr(now, 'mmddHHMM');
if ~exist(save_path)
	mkdir(save_path)
end
diary([save_path, filesep, 'main.', t, '.log']);
diary on;

fprintf('noise: %s \n', noise_type);
fprintf('snr: %s \n', num2str(snr));
fprintf('win_len: %d \n', win_len);
fprintf('win_shift: %d \n', win_shift);
fprintf('file_per_batch: %d\n', file_per_batch);
fprintf('hidden_layer_struct: %s\n', num2str(hidden_layer_struct));
fprintf('learn_rate: %f \n', initial_learn_rate);
fprintf('adjacent_frame: %d \n', adjacent_frame);
fprintf('mini_batch_size: %d \n', mini_batch_size);
fprintf('-------------------------------------------------------\n')

if ~read_from_list

	speech_file_list = dir([speech_path, filesep, '*.wav']);
	num_files = numel(speech_file_list);

	% get validation file list from the beginning
	len_validation = round(num_files*validation_percentage/100);
	if ~len_validation
		error('validation list is empty');
	end
	list_validation = speech_file_list(1:len_validation);

	% get testing file list from the end
	len_testing = round(num_files*testing_percentage/100);
	if ~len_testing
		error('testing list is empty');
	end
	list_testing = speech_file_list(num_files-len_testing+1:num_files);

	% get the training file list in the middle
	list_training = speech_file_list(len_validation+1:num_files-len_testing);
	len_training = numel(list_training);
	if ~len_training
		error('training list is empty');
	end

else

	fid_train_txt = fopen(train_list_path);
	fid_test_txt = fopen(test_list_path);

	tmp = textscan(fid_train_txt, "%s");
    num_files = length(tmp{1});
    for i=1:num_files
        speech_file_list(i).name = tmp{1}(i);
    end

	len_validation = round(num_files*validation_percentage/100);
	if ~len_validation
		error('validation list is empty');
	end
	list_validation = speech_file_list(1:len_validation);

	list_training = speech_file_list(len_validation+1:num_files);
	len_training = numel(list_training);
	if ~len_training
		error('training list is empty');
	end

	tmp = textscan(fid_test_txt, "%s");
	len_testing = length(tmp{1});
    for i=1:len_testing
        list_testing(i).name = tmp{1}(i);
    end

	if ~len_testing
		error('testing list is empty');
	end

end

validation_wav = cell(len_validation, 1);
% read validation files
fprintf('reading validation .wav files...');
for validation_idx=1:len_validation
	[validation_wav{validation_idx}, fs0] = audioread(char(strcat(...
			speech_path, filesep, list_validation(validation_idx).name)));
	% resample to 16k Hz
	validation_wav{validation_idx} = resample(...
				validation_wav{validation_idx}, 16e3, fs0);
end
fprintf('...done\n');

% read noise files
fprintf('reading noise .wav files...');
noise_wav=cell(numel(noise_type),1);
for noise_idx=1:numel(noise_type)
	[noise_wav{noise_idx}, fs0] = audioread(char(strcat(...
				noise_path, filesep, char(noise_type(noise_idx)), '.wav')));
	noise_wav{noise_idx} = resample(noise_wav{noise_idx}, 16e3, fs0);
end
fprintf('...done\n')

% add noise to validation, get the feature and labels
fprintf('obtaining validation data...');
validation_feat = [];
validation_label = [];
for snr_idx=1:numel(snr)
	for noise_idx=1:numel(noise_type)
		cur_snr = snr(snr_idx);
		cur_noise = noise_wav{noise_idx};
		for validation_idx=1:len_validation
			validation_noisy = gen_mix(...
							validation_wav{validation_idx},...
							cur_noise,...
							cur_snr);

			[tmp_feat, tmp_label] = get_training_data(...
								validation_wav{validation_idx},...
								validation_noisy,...
								'ARastaplpMfccGf',...
								win_len,...
								win_shift,...
								fs,...
								useFixedScaFac,...
								sca_fac);
			validation_feat = [validation_feat; tmp_feat'];
			validation_label = [validation_label; tmp_label'];
		end
	end
end
fprintf('...done\n');
validation_feat = single(validation_feat);
feat_len = size(validation_feat, 2);
%validation_feat_win = win_buffer(validation_feat, adjacent_frame);

%validation_feat_win = win_buffer(validation_feat, adjacent_frame);

% initalize the DNN
%layer = gen_DNN_layers(feat_len, win_len, adjacent_frame);
% initalize the training option
%options = gen_training_options(initial_learn_rate, every_train_step, mini_batch_size);
%cur_learn_rate = initial_learn_rate;

% intialize the net
if isempty(checkpoint_path_net)
	net = gen_net(feat_len, ...
					win_len, ...
					adjacent_frame, ...
					hidden_layer_struct, ...
					useInputNormalization, ...
					isGPU);
else
	load(checkpoint_path_net);
end

% initialize the optimizer
if isempty(checkpoint_path_optimizer)
	optimizer = gen_training_optimizer(...
								initial_learn_rate, ...
								every_train_step, ...
								mini_batch_size, ...
								isGPU);
else
	load(checkpoint_path_optimizer);
end

fprintf('-------------------------------------------------------\n')
for training_iter=1:total_train_steps

	fprintf('Training. %d of %d\n', training_iter, total_train_steps);

	single_batch_size = min(file_per_batch, len_training);
	% randomly select single_batch_size files from training list
	batch_list_idx = randi([1, length(list_training)],...
					 		[1, single_batch_size]);
	batch_list = list_training(batch_list_idx);

	fprintf('obtaining training data...')
	if isempty(load_training_data)
		% read training files
		batch_wav = cell(single_batch_size, 1);
		for training_idx=1:single_batch_size
			[batch_wav{training_idx}, fs0] = audioread(char(strcat(speech_path, filesep,...
												batch_list(training_idx).name)));
			% resample to 16k Hz
			batch_wav{training_idx} = resample(batch_wav{training_idx}, 16e3, fs0);
		end
		batch_feat = [];
		batch_label = [];
		for snr_idx=1:numel(snr)
			for noise_idx=1:numel(noise_type)
				cur_snr = snr(snr_idx);
				cur_noise = noise_wav{noise_idx};
				for training_idx=1:single_batch_size
					training_noisy = gen_mix(...
									batch_wav{training_idx},...
									cur_noise,...
									cur_snr);
					[tmp_feat, tmp_label] = get_training_data(...
										batch_wav{training_idx}, ...
										training_noisy, ...
										'ARastaplpMfccGf',...
										win_len,...
										win_shift,...
										fs,...
										useFixedScaFac,...
										sca_fac);
					batch_feat = [batch_feat; tmp_feat'];
					batch_label = [batch_label; tmp_label'];
				end
			end
		end
		batch_feat = single(batch_feat);
		if save_training_data
			save([save_path, filesep, 'training_data.mat'], 'batch_feat', 'batch_label');
		end
	else
		load(load_training_data);
	end
	fprintf('...done\n')

	if useInputNormalization
		[batch_feat_norm, net.norm_mu, net.norm_std] = mean_var_norm(batch_feat);
		batch_feat_win = win_buffer(batch_feat_norm, adjacent_frame);
	else
		batch_feat_win = win_buffer(batch_feat, adjacent_frame);
	end
	
	%{
	if training_iter == 1
		net = trainNetwork(batch_feat_win', ...
										batch_label', ...
										layer, ...
										options);
		layer = net.Layers;
	else
		net = trainNetwork(batch_feat_win', ...
										batch_label', ...
										layer, ...
										options);
		layer = net.Layers;

	end
	%cur_learn_rate = train_info.BaseLearnRate(end);
	%cur_learn_rate = cur_learn_rate * learn_rate_decay_fac;
	
	options = gen_training_options(cur_learn_rate, ...
									 every_train_step, ...
									 mini_batch_size);
	%}
	fprintf('%-20s %-20s %s\n', 'Epoch', 'Cost', 'validtion mse');

	if useInputNormalization
		validation_feat_norm = mean_var_norm_testing(...
									validation_feat, ...
									net.norm_mu, ...
									net.norm_std);
		validation_feat_win = win_buffer(validation_feat_norm, adjacent_frame);
	else
		validation_feat_win = win_buffer(validation_feat, adjacent_frame);
	end
	[net, optimizer] = train_net(batch_feat_win, ...
									batch_label, ...
									validation_feat_win, ...
									validation_label, ...
									net, ...
									optimizer);

	if ~mod(training_iter, checkpoint_save_steps)
		save_checkpoint_path = [save_path, filesep, 'checkpoint'];
		if ~exist(save_checkpoint_path)
			mkdir(save_checkpoint_path);
		end
		save([save_checkpoint_path, filesep, 'checkpoint_step', ...
						num2str(training_iter), '_net.mat'], 'net');
		save([save_checkpoint_path, filesep, 'checkpoint_step', ...
						num2str(training_iter), '_optimizer.mat'], 'optimizer');
	end
	fprintf('-------------------------------------------------------\n')
end

% generate test samples for evaluation
save_idx = 1;
fprintf('Testing. Total %d files...', numel(list_testing));

eval = [];
eval.pesq = zeros(numel(list_testing)*numel(snr)*numel(noise_type),3);
eval.ssnr = eval.pesq;
eval.stoi = eval.pesq;

eval.avg_pesq = zeros(numel(snr)*numel(noise_type), 3);
eval.avg_stoi = eval.avg_pesq;
eval.avg_ssnr = eval.avg_pesq;

for testing_item=1:numel(list_testing)
	[testing_clean, fs0] = audioread(char(strcat(speech_path, filesep,...
							 list_testing(testing_item).name)));
	testing_clean = resample(testing_clean, 16e3, fs0);
	for snr_idx=1:numel(snr)
		for noise_idx=1:numel(noise_type)
			cur_snr = snr(snr_idx);
			cur_noise = noise_wav{noise_idx};
			testing_noisy = gen_mix(...
							testing_clean, ...
							cur_noise, ...
							cur_snr);
			[testing_feat, testing_label] = get_training_data(...
									testing_clean, ...
									testing_noisy, ...
									'ARastaplpMfccGf',...
									win_len,...
									win_shift,...
									fs,...
									useFixedScaFac,...
									sca_fac);
			testing_label = testing_label';
			testing_feat = testing_feat';

			if useInputNormalization
				testing_feat_norm = mean_var_norm_testing(...
													testing_feat, ...
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
											testing_noisy, ...
											fs, ...
											win_len, ...
											win_shift);
			testing_ideal = wav_synthesis(testing_label', ...
										testing_noisy, ...
										fs, ...
										win_len, ...
										win_shift);

			min_len = min(min(length(testing_ideal), length(testing_ideal)), length(testing_noisy));
			testing_noisy = testing_noisy(1:min_len);
			testing_noisy = testing_noisy(:);
			testing_clean = testing_clean(1:min_len);
			testing_clean = testing_clean(:);
			testing_estimated = testing_estimated(1:min_len);
			testing_estimated = testing_estimated(:);

			eval.pesq(save_idx, 1) = pesq(testing_clean, testing_noisy, fs);
			eval.pesq(save_idx, 2) = pesq(testing_clean, testing_ideal, fs);
			eval.pesq(save_idx, 3) = pesq(testing_clean, testing_estimated, fs);

			eval.ssnr(save_idx, 1) = snrseg(testing_noisy, testing_clean, fs);
			eval.ssnr(save_idx, 2) = snrseg(testing_ideal, testing_clean, fs);
			eval.ssnr(save_idx, 3) = snrseg(testing_estimated, testing_clean, fs);

			eval.stoi(save_idx, 1) = stoi(testing_clean, testing_noisy, fs);
			eval.stoi(save_idx, 2) = stoi(testing_clean, testing_ideal, fs);
			eval.stoi(save_idx, 3) = stoi(testing_clean, testing_estimated, fs);

			avg_idx = noise_idx + numel(snr) * (snr_idx - 1);
			eval.avg_pesq(avg_idx, 1) = eval.avg_pesq(avg_idx, 1) + eval.pesq(save_idx, 1);
			eval.avg_pesq(avg_idx, 2) = eval.avg_pesq(avg_idx, 2) + eval.pesq(save_idx, 2);
			eval.avg_pesq(avg_idx, 3) = eval.avg_pesq(avg_idx, 3) + eval.pesq(save_idx, 3);

			eval.avg_ssnr(avg_idx, 1) = eval.avg_ssnr(avg_idx, 1) + eval.ssnr(save_idx, 1);
			eval.avg_ssnr(avg_idx, 2) = eval.avg_ssnr(avg_idx, 2) + eval.ssnr(save_idx, 2);
			eval.avg_ssnr(avg_idx, 3) = eval.avg_ssnr(avg_idx, 3) + eval.ssnr(save_idx, 3);

			eval.avg_stoi(avg_idx, 1) = eval.avg_stoi(avg_idx, 1) + eval.stoi(save_idx, 1);
			eval.avg_stoi(avg_idx, 2) = eval.avg_stoi(avg_idx, 2) + eval.stoi(save_idx, 2);
			eval.avg_stoi(avg_idx, 3) = eval.avg_stoi(avg_idx, 3) + eval.stoi(save_idx, 3);

			save_path_estimated = char(strcat(save_path, filesep, 'testing', filesep, 'estimated', filesep, num2str(snr(snr_idx)), filesep, noise_type(noise_idx)));
			save_path_clean = char(strcat(save_path, filesep, 'testing', filesep, 'clean', filesep, num2str(snr(snr_idx)), filesep, noise_type(noise_idx)));
			save_path_ideal = char(strcat(save_path, filesep, 'testing', filesep, 'ideal', filesep, num2str(snr(snr_idx)), filesep, noise_type(noise_idx)));
			save_path_mix = char(strcat(save_path, filesep, 'testing', filesep, 'mix', filesep, num2str(snr(snr_idx)), filesep, noise_type(noise_idx)));
			if ~exist(save_path_estimated)
				mkdir(save_path_estimated);
			end
			if ~exist(save_path_clean)
				mkdir(save_path_clean);
			end
			if ~exist(save_path_ideal)
				mkdir(save_path_ideal);
			end			
			if ~exist(save_path_mix)
				mkdir(save_path_mix);
			end			
			audiowrite( ...
				char(strcat(save_path_estimated, filesep, list_testing(testing_item).name)),...
				testing_estimated, ...
				fs);
			audiowrite( ...
				char(strcat(save_path_clean, filesep, list_testing(testing_item).name)),...
				testing_clean, ...
				fs);
			audiowrite( ...
				char(strcat(save_path_ideal, filesep, list_testing(testing_item).name)),...
				testing_ideal, ...
				fs);
			testing_noisy = testing_noisy / max(abs(testing_noisy));
			audiowrite( ...
				char(strcat(save_path_mix, filesep, list_testing(testing_item).name)),...
				testing_noisy, ...
				fs);
			save_idx = save_idx + 1;
		end
	end
end
eval.avg_pesq = eval.avg_pesq ./ (numel(list_testing) * numel(snr) * numel(noise_type));
eval.avg_stoi = eval.avg_stoi ./ (numel(list_testing) * numel(snr) * numel(noise_type));
eval.avg_ssnr = eval.avg_ssnr ./ (numel(list_testing) * numel(snr) * numel(noise_type));

fprintf('...done\n');

save([save_path, filesep, 'testing', filesep, 'net.mat'], 'net');
save([save_path, filesep, 'testing', filesep, 'optimizer.mat'], 'optimizer');

fid=fopen([save_path, filesep, 'testing', filesep, 'info.txt'], 'w');
fprintf(fid, 'noise: %s \n', noise_type);
fprintf(fid, 'snr: %.0f \n', snr);
fprintf(fid, 'list of testing wav files: \n');
for testing_item=1:numel(list_testing)
	fprintf(fid, char(strcat(list_testing(testing_item).name, '\n')));
end

% print the evaluation result
fprintf(fid, '\n');
print_eval_res(fid, eval, 'mix', noise_type, snr, list_testing, 0);
print_eval_res(fid, eval, 'ideal', noise_type, snr, list_testing, 0);
print_eval_res(fid, eval, 'estimated', noise_type, snr, list_testing, 0);

print_eval_res(fid, eval, 'mix', noise_type, snr, list_testing, 1);
print_eval_res(fid, eval, 'ideal', noise_type, snr, list_testing, 1);
print_eval_res(fid, eval, 'estimated', noise_type, snr, list_testing, 1);

fclose(fid);
diary off;