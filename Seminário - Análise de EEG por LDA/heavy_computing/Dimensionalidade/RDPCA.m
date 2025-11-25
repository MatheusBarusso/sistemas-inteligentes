function [Z, V, lambda, explained] = RDPCA(X, varargin)

 % média móvel, para evitar valores com NaN
 for i = 1:size(X,2)
    col = X(:, i);
    col(isnan(col)) = mean(col(~isnan(col)));
    X(:, i) = col;
end


    % ======================
    % 1) Centralização dos dados
    % ======================
    X = double(X);                 % Garante formato double
    mu = mean(X, 1);               % Média de cada coluna (feature)
    Xc = X - mu;                   % Dados centralizados

    % ======================
    % 2) Matriz de covariância
    % ======================
    C = cov(Xc);

    % ======================
    % 3) Autovalores e Autovetores
    % ======================
    [V, D] = eig(C);               % Decomposição espectral
    [lambda, idx] = sort(diag(D), 'descend');  % Ordena autovalores
    V = V(:, idx);                 % Reorganiza autovetores

    % ======================
    % 4) Variância explicada
    % ======================
    totalVar = sum(lambda);
    explained = cumsum(lambda) / totalVar * 100;

    % ======================
    % 5) Seleção do número de componentes (k)
    % ======================
    k = length(lambda);  % Padrão: mantém tudo

    if ~isempty(varargin)
        if strcmpi(varargin{1}, 'k')
            k = varargin{2};
        elseif strcmpi(varargin{1}, 'var')
            k = find(explained >= varargin{2}, 1, 'first');
        end
    end

    V = V(:, 1:k);            % Mantém os k componentes principais
    Z = Xc * V;               % Projeção dos dados

end