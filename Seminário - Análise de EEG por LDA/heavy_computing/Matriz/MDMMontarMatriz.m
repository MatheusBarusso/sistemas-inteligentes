function [X, feature_names] = MDMMontarMatriz(varargin)


feature_names = {};
X_list = {};  % lista temporária para concatenar

for i = 1:length(varargin)
    data = varargin{i};
    
    if isstruct(data)
        fields = fieldnames(data);
        for f = 1:length(fields)
            val = data.(fields{f});
            sz = size(val);
            
            % Supondo formato [Sujeito x Canal x Trial x Epoca]
            if numel(sz) == 4
                [NumS, NumCh, NumTr, NumEp] = deal(sz(1), sz(2), sz(3), sz(4));
                NumSamples = NumS * NumTr * NumEp;
                
                % Inicializar vetor para armazenar medida
                X_field = zeros(NumSamples, NumCh);
                
                row_idx = 1;
                for s = 1:NumS
                    for tr = 1:NumTr
                        for ep = 1:NumEp
                            X_field(row_idx, :) = squeeze(val(s,:,tr,ep));
                            row_idx = row_idx + 1;
                        end
                    end
                end
                
                X_list{end+1} = X_field;
                
                % Nomes das features
                for ch = 1:NumCh
                    feature_names{end+1} = sprintf('%s_ch%d', fields{f}, ch);
                end
            else
                % Para structs 2D ou outros formatos
                X_list{end+1} = val;
                if isvector(val)
                    feature_names{end+1} = fields{f};
                else
                    for col = 1:size(val,2)
                        feature_names{end+1} = sprintf('%s_col%d', fields{f}, col);
                    end
                end
            end
        end
        
    elseif isnumeric(data)
        sz = size(data);
        if sz(1) ~= 1 && sz(2) ~= 1
            X_list{end+1} = data;
            % Nomes genéricos
            for col = 1:sz(2)
                feature_names{end+1} = sprintf('num%d', col);
            end
        else
            X_list{end+1} = data(:);  % vetorizar
            feature_names{end+1} = 'vector';
        end
    else
        error('Tipo de dado não suportado');
    end
end

% Concatenar horizontalmente todas as medidas
X = horzcat(X_list{:});

end