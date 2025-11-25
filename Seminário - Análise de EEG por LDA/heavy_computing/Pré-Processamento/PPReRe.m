function EEG_ref = PPReRe(EEG_data, chan_names, chan_coords, method)

[S, C, N, T] = size(EEG_data);
EEG_ref = EEG_data; % inicialização

switch lower(method)
    case 'car'
        % Média Comum (CAR)
        mean_ch = mean(EEG_data, 2); % média por canal
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - mean_ch;
        end
    otherwise
        error('Método desconhecido. Use: "CAR".');
end

end
