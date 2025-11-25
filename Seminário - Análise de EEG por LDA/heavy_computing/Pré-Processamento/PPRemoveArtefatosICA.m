function [EEG_clean, S_all, W_all, removed_all] = PPRemoveArtefatosICA(EEG_data)
[NumSujeito, NumCan, NumAmostras, NumTrial] = size(EEG_data);

% Inicializações
EEG_clean = zeros(size(EEG_data));
S_all = cell(NumSujeito, NumTrial);
W_all = cell(NumSujeito, NumTrial);
removed_all = cell(NumSujeito, NumTrial);

fprintf('\nICA para remoção de artefatos processando\n');

for s = 1:NumSujeito
    fprintf('\nSujeito %d/%d', s, NumSujeito);
    
    for tr = 1:NumTrial
        fprintf('.');
        
        % Extrair trial (C x N)
        X = squeeze(EEG_data(s, :, :, tr));
        
        % Verifica se o sinal é válido
        if any(isnan(X(:))) || all(X(:) == 0)
            EEG_clean(s,:,:,tr) = 0;
            continue;
        end

        % Centralizar dados (zero mean por canal)
        X = X - mean(X,2);

        % Whitening
        [C, N] = size(X);
        E = cov(X');
        [U, D] = eig(E);
        D_inv_sqrt = diag(1./sqrt(diag(D) + eps));  % evitar divisão por zero
        X_white = D_inv_sqrt * U' * X;

        % ICA via maximização da kurtosis
        W = randn(C,C);
        maxIter = 500; tol = 1e-6;
        for iter = 1:maxIter
            W_old = W;
            Y = W*X_white;
            gY = Y.^3;
            W = (gY*Y')/N - 3*eye(C)*W;
            % Ortogonalização
            [Uo, ~, Vo] = svd(W);
            W = Uo*Vo';
            % Convergência
            if max(abs(abs(diag(W*W_old'))-1)) < tol
                break;
            end
        end

        % Componentes independentes
        S = W * X_white;

        % Detecção automática de artefatos (desvio padrão alto)
        std_IC = std(S,0,2);
        threshold = mean(std_IC) + 2*std(std_IC);
        removed_components = find(std_IC > threshold);

        % Zerar componentes artefatuais
        S_clean = S;
        S_clean(removed_components,:) = 0;

        % Reconstrução do EEG limpo
        EEG_clean(s,:,:,tr) = pinv(W) * S_clean;

        % Armazenar resultados
        S_all{s,tr} = S;
        W_all{s,tr} = W;
        removed_all{s,tr} = removed_components;
    end
end

fprintf('\nICA concluído\n');
end