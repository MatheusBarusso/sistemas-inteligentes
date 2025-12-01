import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
def carregar_e_preprocessar_dados(caminho_arquivo):

    df = pd.read_csv(caminho_arquivo)
    df_processed = df.copy() #copia p processamento
    df_processed = df_processed.dropna() #valores ausentes (se tiver)
    
    le_sexo = LabelEncoder()
    le_escola = LabelEncoder()
    df_processed['Sexo_encoded'] = le_sexo.fit_transform(df_processed['Sexo'])
    df_processed['Escola_encoded'] = le_escola.fit_transform(df_processed['Escola'])
    
    return df, df_processed, le_sexo, le_escola

# MÉTODO DO COTOVELO
def metodo_cotovelo(df_processed, max_k=10):
    # Selecionar features
    features = ['Coeficiente', 'Enem', 'Escola_encoded']
    X = df_processed[features]
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calcular soma das distâncias quadradas para diferentes k
    inercias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
    
    # Plotar método do cotovelo
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inercias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Soma das Distâncias Quadradas', fontsize=12)
    plt.title('Método do Cotovelo para Determinação do K Ideal', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(K_range)
    
    for i, (k, inercia) in enumerate(zip(K_range, inercias)):
        plt.annotate(f'{inercia:.1f}', (k, inercia), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()
    
    return X_scaled, scaler, features

# K-MEANS
def aplicar_kmeans(X_scaled, df_processed, n_clusters=3):
    """
    Aplica K-Means com o número selecionado de clusters
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_processed['Cluster'] = kmeans.fit_predict(X_scaled)
    
    return kmeans

# ANÁLISE DOS CLUSTERS
def analisar_clusters(df, df_processed):
    
    for cluster in sorted(df_processed['Cluster'].unique()):
        
        print(f"\n\nCLUSTER {cluster}")
        
        cluster_data = df_processed[df_processed['Cluster'] == cluster]
        
        print(f"\nTamanho: {len(cluster_data)} alunos ({len(cluster_data)/len(df_processed)*100:.1f}%)")
        
        print(f"\nCoeficiente:")
        print(f"  Média: {cluster_data['Coeficiente'].mean():.2f}")
        print(f"  Mediana: {cluster_data['Coeficiente'].median():.2f}")
        print(f"  Desvio padrão: {cluster_data['Coeficiente'].std():.2f}")
        
        print(f"\nNota ENEM:")
        print(f"  Média: {cluster_data['Enem'].mean():.2f}")
        print(f"  Mediana: {cluster_data['Enem'].median():.2f}")
        print(f"  Desvio padrão: {cluster_data['Enem'].std():.2f}")
        
        print(f"\nDistribuição por Sexo:")
        sexo_dist = cluster_data['Sexo'].value_counts()
        for sexo, count in sexo_dist.items():
            print(f"  {sexo}: {count} ({count/len(cluster_data)*100:.1f}%)")
        
        print(f"\nDistribuição por Escola:")
        escola_dist = cluster_data['Escola'].value_counts()
        for escola, count in escola_dist.items():
            print(f"  {escola}: {count} ({count/len(cluster_data)*100:.1f}%)")
    

def main():
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    caminho_arquivo = os.path.join(diretorio_script, 'alunos_engcomp.csv')
    
    try:
        df, df_processed, le_sexo, le_escola = carregar_e_preprocessar_dados(caminho_arquivo)
        
        X_scaled, scaler, features = metodo_cotovelo(df_processed, max_k=10)
        
        n_clusters = int(input("\nInsira o número de clusters: "))
        
        kmeans = aplicar_kmeans(X_scaled, df_processed, n_clusters)
        
        analisar_clusters(df, df_processed)
        
    except FileNotFoundError:
        print(f"\nErro: Arquivo 'alunos_engcomp.csv' não encontrado no diretório.")
    except Exception as e:
        print(f"\nErro durante a análise: {str(e)}")

if __name__ == "__main__":
    main()