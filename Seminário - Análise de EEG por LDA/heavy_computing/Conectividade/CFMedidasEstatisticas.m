function features = CFMedidasEstatisticas(conn_all, thresh)

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
            
            % MEDIDAS GLOBAIS
            mean_conn = mean(M(:));
            var_conn  = var(M(:));

            % Evitar log(0) na entropia
            M = M(:);
            M = M - min(M);         % remove valores negativos, se existirem
            M = M + eps;            % evita zeros

            % Normalização correta
            M = M / sum(M);

            % Entropia de Shannon
            entropy_M = -sum(M .* log(M));

            max_conn = max(M(:));
            %min_conn = min(M(:));
            max_conn = max(M(:));
            min_conn = min(M(:));
            
            
            % ORGANIZAR VETOR DE FEATURES
            feat_vec = [mean_conn, var_conn, entropy_M, ...
                        max_conn, min_conn];
            
            % Concatenar no dataset final
            features = [features; feat_vec];
            
        end
    end
end

end