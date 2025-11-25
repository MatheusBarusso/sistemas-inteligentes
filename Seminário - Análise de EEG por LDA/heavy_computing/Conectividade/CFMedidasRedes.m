function features = CFMedidasRedes(conn_all)

if nargin < 2
    thresh = 0.5; % limiar padrão para grau
end

NumSuj = size(conn_all,1);
NumTr  = size(conn_all,2);
NumEp  = size(conn_all,3);

% Exemplo: armazenar em célula ou matriz
features = [];  % acumulador de vetores

for s = 1:NumSuj
    for tr = 1:NumTr
        for ep = 1:NumEp
            
            M = conn_all{s,tr,ep};   % matriz NxN de conectividade
            
            if isempty(M)
                continue;
            end
            
            % MEDIDAS LOCAIS
            degree   = sum(M > thresh, 2); % vetor Nx1
            strength = sum(M, 2);          % vetor Nx1
            
            mean_degree   = mean(degree);
            mean_strength = mean(strength);
            
            
            % ORGANIZAR VETOR DE FEATURES
            feat_vec = [mean_degree, mean_strength];
            
            % Concatenar no dataset final
            features = [features; feat_vec];
            
        end
    end
end

end