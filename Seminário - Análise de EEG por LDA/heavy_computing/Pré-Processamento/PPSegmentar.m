function [EEG_epochs,num_epochs] = PPSegmentar(EEG_data, fs, window_sec, overlap_sec)
[S, C, N, T] = size(EEG_data);
win_len = round(window_sec * fs);
step = round((window_sec - overlap_sec) * fs);
num_epochs = floor((N - win_len)/step) + 1;

EEG_epochs = zeros(S, C, win_len, num_epochs, T);

for s = 1:S
    for tr = 1:T
        start_idx = 1;
        for e = 1:num_epochs
            idx = start_idx:(start_idx + win_len - 1);
            EEG_epochs(s,:,:,e,tr) = EEG_data(s,:,idx,tr);
            start_idx = start_idx + step;
        end
    end
end

end