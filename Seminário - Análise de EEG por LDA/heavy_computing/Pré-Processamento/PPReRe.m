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

    case 'cz'
        % Referência em Cz
        idxCz = find(strcmpi(chan_names,'Cz'),1);
        if isempty(idxCz)
            error('Canal Cz não encontrado.');
        end
        Cz_sig = EEG_data(:,idxCz,:,:);
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - Cz_sig;
        end

    case 'mastoides'
        % Referência A1/A2
        idxA1 = find(strcmpi(chan_names,'A1'),1);
        idxA2 = find(strcmpi(chan_names,'A2'),1);
        if isempty(idxA1) || isempty(idxA2)
            error('Canais A1/A2 não encontrados.');
        end
        ref_sig = (EEG_data(:,idxA1,:,:) + EEG_data(:,idxA2,:,:))/2;
        for ch = 1:C
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - ref_sig;
        end

    case 'laplaciano'
        % Laplaciano espacial (canais vizinhos)
        if isempty(chan_coords)
            error('Para Laplaciano, chan_coords deve ser fornecido.');
        end
        % Distância euclidiana entre canais
        dist = pdist2(chan_coords, chan_coords);
        for ch = 1:C
            % vizinhos próximos (dist < threshold)
            threshold = 0.2; % ajustar conforme escala
            neighbors = find(dist(ch,:) > 0 & dist(ch,:) <= threshold);
            if isempty(neighbors)
                continue;
            end
            mean_neighbors = mean(EEG_data(:,neighbors,:,:),2);
            EEG_ref(:,ch,:,:) = EEG_data(:,ch,:,:) - mean_neighbors;
        end

    otherwise
        error('Método desconhecido. Opções: CAR, Cz, Mastoides, Laplaciano.');
end

end
