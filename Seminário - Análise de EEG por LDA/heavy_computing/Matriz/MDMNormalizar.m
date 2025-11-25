function X_norm = MDMNormalizar(X, method)

switch lower(method)
    case 'zscore'
        X_norm = (X - mean(X,1)) ./ std(X,[],1);
    case 'minmax'
        X_min = min(X,[],1);
        X_max = max(X,[],1);
        X_norm = (X - X_min) ./ (X_max - X_min);
    case 'robust'
        X_med = median(X,1);
        X_iqr = iqr(X,1);
        X_norm = (X - X_med) ./ X_iqr;
    otherwise
        error('Método de normalização inválido. Use "zscore", "minmax" ou "robust".');
end

end