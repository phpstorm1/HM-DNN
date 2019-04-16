function [ training_label ] = get_harm_labels( voice_sig, ...
												 mix_sig, ...
												 win_len, ...
												 win_shift, ...
												 fs, ...
												 use_fixed_sca, ...
												 sca_fac )

%Returns a harmonic training label matrix for a voiced speech
%	Input:
%	voice_sig: single-channel clean speech
%	mix_sig: single-channel noisy speech
%	win_len: window length for framing
%	win_shift: shifts between window
%	fs: sampling frequency
%	use_fixed_sca: flag inficating whether fixed scaling factor is used
%	sca_fac: fiexed scaling factor for residual
%	Output:
%	training_label: the magnitude of STFT

threshold_snr = 0.9;
win_fun = sqrt(hanning(win_len, 'periodic'));

voice_sig = voice_sig(:)';
voice_sig = voice_sig / max(abs(voice_sig));
mix_sig = mix_sig / max(abs(mix_sig));

harm_mask = get_har_mask(voice_sig, fs, win_len, win_shift);
spch_frame = enframe(voice_sig, win_fun, win_shift);
freq_frame = fft(spch_frame')';

mix_frame = enframe(mix_sig, win_fun, win_shift);
mix_freq = abs(fft(mix_frame')');

frame_num = size(mix_freq, 1);
frame_snr = zeros(frame_num, 1);

for i=1:frame_num
	frame_snr(i) = sum(abs(freq_frame(i,:))) / sum(mix_freq(i,:));
end
mask_sca = frame_snr > threshold_snr;

resi_mask = ones(size(spch_frame)) - harm_mask;

resi_freq = freq_frame .* resi_mask;
ada_sca = zeros(size(resi_mask));
for i = 1:size(resi_mask,1)
	c_max = abs(max(resi_freq(i,:)));
	if ~c_max
		continue;
	end
	c_min = abs(min(resi_freq(i,:)));
	ada_sca(i,:) = sqrt(abs(resi_freq(i,:)) ./ (c_max-c_min));
	% add snr-aware-adaptive scaling
	if mask_sca(i)
		ada_sca(i,:) = 1;
	end
end

if use_fixed_sca
	label_freq = harm_mask.* freq_frame + sca_fac * freq_frame .* resi_mask;
else
	label_freq = harm_mask.* freq_frame + ada_sca .* resi_freq;
end
	
% add gobal_sca (kinda like 'SNR aware')
%gobal_sca = sqrt(sum(voice_sig.^2)/sum(mix_sig.^2));
%label_freq = harm_mask.* freq_frame + gobal_sca * ada_sca .* resi_freq;

amp_harm = abs(label_freq);
[num_frame, len_frame] = size(spch_frame);
rep = win_len/2+1;

% used only train the amplitude
training_label = amp_harm(:,1:rep);
training_label = training_label';

end

