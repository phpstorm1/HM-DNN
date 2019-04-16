function [ harm_mask] = get_har_mask( speech_wav, fs, win_len, win_shift )
%Return a binary matrix with every component indicating if the frequency
%bin is harmonic frequency or not. Note that the DFT length is set by the
%input parameter win_len.
%
% Input:
% speech_wav: a speech one-demension speech signal
% fs: sampling frequency
% win_len: window length
% win_shift: window shift
%
% Ouput:
% harmonic_freq_mask: a num_frame*num_sample binary matrix, with 1 representing
% the current frequency bin is the harmonic frequency, and vice versa.

THRES_PROB = 0.15;
freq_res = fs/win_len;
frame_speech=enframe(speech_wav, win_len, win_shift);

% the hop for pitch estimation is set to win_shift
[pitch, time, prob] = fxpefac(speech_wav, fs, win_shift/fs);
time_idx = floor(time.*fs);
[num_frame, num_sample] = size(frame_speech);
pitch_frame = zeros(1,num_frame);
prob_frame = zeros(1, num_frame);
time_frame = zeros(1,num_frame);
harm_mask = zeros(num_frame, num_sample);

% using the result obtained from pefac to estimate the occurance of pitch
% among every single frame
for i = 1:num_frame
    % sum up the estimated pitch and probability of activity speech if they
    % are in the current frame
    for j = 1:length(time_idx)
        if time_idx(j) >= (i-1)*win_shift && time_idx(j) <= (i-1)*win_shift+win_len
            pitch_frame(i) = pitch_frame(i) +  pitch(j);
            prob_frame(i) = prob_frame(i) + prob(j);
            time_frame(i) = time_frame(i) + 1;
        end
    end
    if time_frame(i) 
        % weight the ptich and probability by the number of the occurance
        pitch_frame(i) = pitch_frame(i)/time_frame(i);
        prob_frame(i) = prob_frame(i)/time_frame(i);
        if prob_frame(i) >= THRES_PROB
            % highest harmonic frequency cannot exceed Nyquist frequency
            n_harm = floor(fs*0.5/pitch_frame(i));
            for k = 1:n_harm
                % first point represents DC, i.e., frequency=0
                freq_idx = round(k*pitch_frame(i)/freq_res+1);
                harm_mask(i,freq_idx) = 1;
            end
        end
    end
end

end