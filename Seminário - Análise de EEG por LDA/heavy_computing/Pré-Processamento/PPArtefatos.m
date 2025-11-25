function [bad_epoch, EEG_clean] = PPArtefatos(EEG_data, ampThreshold)
[NumSujeito, ~, ~, NumTrial] = size(EEG_data);

bad_epoch = false(NumSujeito, NumTrial); % inicializa matriz de artefatos
EEG_clean = EEG_data;                     % cria cópia dos dados

for s = 1:NumSujeito
    for tr = 1:NumTrial
        sinal = squeeze(EEG_data(s,:,:,tr)); % C x N
        % cálculo peak-to-peak por canal
        p2p = max(sinal,[],2) - min(sinal,[],2);
        if any(p2p > ampThreshold)
            bad_epoch(s,tr) = true;           % marca trial como artefato
            EEG_clean(s,:,:,tr) = NaN;        % substitui por NaN
        end
    end
end

end