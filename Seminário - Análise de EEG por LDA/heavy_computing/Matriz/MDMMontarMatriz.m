function [X, feature_names] = MDMMontarMatriz(varargin)
% MDMMONTARMATRIZ - Monta a matriz de features completa (X) a partir de 
% N entradas, lidando com structs 4D (Mineração) e arrays 2D (Conectividade).
%
% ENTRADAS: N structs 4D [Suj x Canal x Epoca x Trial] ou arrays 2D [Amostras x Features]
% SAÍDAS: X [23328 x 337] e feature_names

feature_names = {};
X_list = {};  % lista temporária para concatenar

for i = 1:length(varargin)
    data = varargin{i};

    if isstruct(data)
        % --- PROCESSA STRUCTS 4D (MINERAÇÃO/HJORTH) ---
        fields = fieldnames(data);
        for f = 1:length(fields)
            val = data.(fields{f});
            sz = size(val);

            % Supondo formato [Sujeito x Canal x Epoca x Trial]
            if numel(sz) == 4
                [NumS, NumCh, NumEp, NumTr] = deal(sz(1), sz(2), sz(3), sz(4));

                % O total de Amostras é Sujeito * Epoca * Trial (9 * 81 * 32 = 23328)
                NumSamples = NumS * NumEp * NumTr; 

                % 1. Reordenar: Move a dimensão do Canal (Feature) para o final
                % [Suj x Canal x Epoca x Trial] -> [Suj x Epoca x Trial x Canal]
                val_reordenado = permute(val, [3 4 1 2]); 

                % 2. Achatamento para [NumSamples x NumCh]
                X_field = reshape(val_reordenado, NumSamples, NumCh);

                X_list{end+1} = X_field;

                % Nomes das features
                for ch = 1:NumCh
                    feature_names{end+1} = sprintf('%s_%s_ch%d', fields{f}, fields{f}, ch);
                end

            else % Se a struct for 2D (caso não esperado, mas por segurança)
                error('Formato de struct não reconhecido. Esperado 4D para Mineração.');
            end
        end

    elseif isnumeric(data)
        % --- PROCESSA ARRAYS NUMÉRICOS 2D (PLV/CORRELAÇÃO) ---
        sz = size(data);
        if ndims(data) == 2
            X_list{end+1} = data;

            % Nomes genéricos, ajustando para não conflitar
            base_name = ['feat_array_', num2str(i)];
            for col = 1:sz(2)
                feature_names{end+1} = sprintf('%s_col%d', base_name, col);
            end
        else
            error('Array numérico de entrada não é 2D. Verifique a agregação do PLV/Correlação.');
        end

    else
        warning('Ignorando entrada de tipo desconhecido na coluna %d', i);
    end
end

% CONCATENAÇÃO FINAL
X = horzcat(X_list{:});

end



