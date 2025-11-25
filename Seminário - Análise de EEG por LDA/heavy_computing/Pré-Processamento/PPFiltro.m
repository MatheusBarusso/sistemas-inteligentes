function EEG_data = PPFiltro(EEG_data,fs)
[NumSujeito, NumCan, ~, NumTrial] = size(EEG_data);

f_low = 0.5; f_high = 30; order = 100;
b_bp = fir1(order, [f_low f_high]/(fs/2), 'bandpass');

f0 = 60; Q = 30;
[b_notch, a_notch] = iirnotch(f0/(fs/2), f0/(fs/2)/Q);

f_highpass = 0.5; order_hp = 50;
b_hp = fir1(order_hp, f_highpass/(fs/2), 'high');

for s = 1:NumSujeito
    for ch = 1:NumCan
        for tr = 1:NumTrial

            % Extrair o sinal
            sinal = squeeze(EEG_data(s,ch,:,tr));

            % Passa-banda 0.5–30 Hz
            sinal = filtfilt(b_bp, 1, sinal);

            % Notch 60 Hz
            sinal = filtfilt(b_notch, a_notch, sinal);

            % Passa-alta 0.5 Hz
            sinal = filtfilt(b_hp, 1, sinal);

            % Atualizar diretamente na matriz original
            EEG_data(s,ch,:,tr) = sinal;
        end
    end
end