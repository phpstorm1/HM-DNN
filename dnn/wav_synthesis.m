function [ syn_wav ] = wav_synthesis( label, mix, fs, win_len, win_shift )
%Returns a synthesised speech using the output from NN
% Input:
%	label: output from DNN, matrix containing amplititude and IF (phase)
%	fs: sampling frequency
%	win_len: window length
%	win_shift: shift between windows

win_fun = sqrt(hanning(win_len, 'periodic'));

% use mix phase for generating the waveform
mix_frame = enframe(mix, win_fun, win_shift);

mix_fft = fft(mix_frame')';
mix_phase = angle(mix_fft);

label = label';
num_col_split = size(label,2);
num_frame = size(label, 1);
amp_frame = label(:, 1:num_col_split);

% throw the values below the threshold
thres = 0;
idx = amp_frame<thres;
amp_frame(idx) = 0;

dft_frame = zeros(num_frame, win_len);
wav_frame = zeros(num_frame, win_len);

% for i=1:num_frame
%     for j=1:num_col_split
%         dft_frame(i,j) = amp_frame(i,j)*exp(1j*mix_phase(i,j));
%     end
% end

exp_phase = exp(1j.*mix_phase);
dft_frame(:, 1:num_col_split) = amp_frame(:, 1:num_col_split) .* exp_phase(:, 1:num_col_split);

wav_frame = real(ifft(dft_frame')');
syn_wav = overlapadd(wav_frame, win_fun, win_shift);
syn_wav = syn_wav./max(abs(syn_wav));

end
