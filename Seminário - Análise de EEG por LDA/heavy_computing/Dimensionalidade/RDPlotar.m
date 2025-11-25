function RDPlotar(explained, lambda)

    figure;
    
    % Plotar variância explicada
    subplot(1, 2, 1);
    bar(explained, 'FaceAlpha', 0.7);
    xlabel('Componente Principal');
    ylabel('Variância Explicada (%)');
    title('Variância Explicada por Componente');
    grid on;

    % Autovalores
    if nargin > 1
        subplot(1, 2, 2);
        plot(lambda, '-o', 'LineWidth', 1.5);
        xlabel('Componente Principal');
        ylabel('Autovalor');
        title('Scree Plot (Autovalores)');
        grid on;
    end

end