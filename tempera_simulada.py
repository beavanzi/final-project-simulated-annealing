import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd
import random
import math
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import time

ano_dos_ativos = 2018
folder = f'resultados_{ano_dos_ativos}'
lista_acoes = ["ABEV3", "ITUB4", "PETR4", "VALE3", "BBDC4", "SANB11"]
dict_retorno_ativo_livre_risco = {2018: 0.065, 2019: 0.055, 2023: 0.1225}

def calcula_retorno_portfolio(portfolio, retornos):
    return np.sum(portfolio * retornos)

def calcula_retorno_periodo_total_acoes(dominio):
    n, m = dominio.shape
    array_retornos = np.zeros(m)
    for j in range(m):
        array_retornos[j] = (dominio[-1][j] - dominio[0][j]) / dominio[0][j]
    return array_retornos

def calcula_covariancia_acoes(variacao_precos):
    covariancia = variacao_precos.cov()
    return covariancia

def calcula_variacao_precos(precos):
    prices_df = pd.DataFrame(precos, columns=['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F'])
    # calcular a covariancia entre os ativos. cada elemento desta matriz representa a covariância (como variam) entre os retornos de dois ativos diferentes. a covariância é uma medida de como dois ativos se movimentam juntos.
    variacao_precos = prices_df / prices_df.shift(1) - 1
    variacao_precos = variacao_precos.dropna()
    return variacao_precos

def calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tamanho_amostra):
    desvio_padrao_por_dia = np.sqrt(np.dot(np.dot(pesos_portfolio.T, covariancia), pesos_portfolio))
    desvio_padrao_total = desvio_padrao_por_dia * np.sqrt(tamanho_amostra)
    return desvio_padrao_total

def funcao_objetivo(pesos_portfolio, precos, taxa_retorno, covariancia, retorno_ativo_livre_risco):
    qtd_dias_variacoes_preco = len(precos)-1
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, qtd_dias_variacoes_preco)
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno) 
    sharpe = (retorno_portfolio - retorno_ativo_livre_risco) / risco_portifolio
    return sharpe


def probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
    sorteio = random.random()
    probabilidade = pow(math.e, (- novo_valor_objetivo - melhor_valor_objetivo) / temperatura)
    return  sorteio < probabilidade


def normaliza_vetor(vetor):
    normalizado = []
    for i in range(len(vetor)):
        normalizado.append((vetor[i] - min(vetor)) / (max(vetor) - min(vetor)));
    return normalizado / np.sum(normalizado)

def distribui_vetor(vetor):
    return vetor / np.sum(vetor)

def tempera_simulada(precos, taxa_retorno, covariancia, retorno_ativo_livre_risco, tot_dias_dados, metricas_iteracoes, temperatura=1000.0, resfriamento=0.9999, passo=0.05):
    n, m = precos.shape
    pesos_portfolio = gerar_primeira_solucao(precos)

    pesos_melhor_portfolio = np.copy(pesos_portfolio)
    melhor_valor_objetivo = funcao_objetivo(pesos_melhor_portfolio, precos, taxa_retorno, covariancia, retorno_ativo_livre_risco)

    contador = 0

    # enquanto a temperatura não foi quase zerada
    while temperatura > 0.1:
        novo_portfolio = np.copy(pesos_portfolio)

        # escolher uma posiçao e alterar ela de acordo com o passo. 
        i = np.random.choice(range(m), size=1, replace=False)
        # direcao é um valor float aleatorio entre -passo e +passo
        direcao = np.random.uniform(-passo, passo)

        if novo_portfolio[i] + direcao > 0:
            novo_portfolio[i] = novo_portfolio[i] + direcao
        else:
            novo_portfolio[i] = novo_portfolio[i] - direcao

        novo_portfolio = normaliza_vetor(novo_portfolio)
        novo_valor_objetivo = funcao_objetivo(novo_portfolio, precos, taxa_retorno, covariancia, retorno_ativo_livre_risco)

        prob_aceita_solucao_ruim = probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura)

        if (novo_valor_objetivo > melhor_valor_objetivo or prob_aceita_solucao_ruim):
            pesos_portfolio = np.copy(novo_portfolio)
            pesos_melhor_portfolio = np.copy(pesos_portfolio)
            melhor_valor_objetivo = novo_valor_objetivo
        
        metricas = gerar_metricas_para_iteracao(contador, taxa_retorno, covariancia, retorno_ativo_livre_risco, pesos_melhor_portfolio, tot_dias_dados, temperatura)
        metricas_iteracoes.append(metricas)
        contador += 1
        temperatura = temperatura * resfriamento
    print('Contador: ', contador)
    return pesos_melhor_portfolio


def gerar_metricas_para_iteracao(contador, taxa_retorno, covariancia, retorno_ativo_livre_risco, pesos_portfolio, tot_dias_dados, temperatura):
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno)
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tot_dias_dados)
    sharpe = (retorno_portfolio - retorno_ativo_livre_risco)/risco_portifolio
    
    return {
        'Iteracao': contador,
        'Retorno': retorno_portfolio,
        'Risco': risco_portifolio,
        'IS': sharpe, 
        'Temperatura': temperatura}
    
def gerar_primeira_solucao(precos):
    # para diminuir a aleatoriedade: calcular os pesos de retorno do dominio normalizados ou usar tudo igual
    n, m = precos.shape
    unitario = [1] * m
    solucao = distribui_vetor(unitario)
    return solucao

def gerar_csv(precos, nome_arquivo):
    n = 10
    
    # Cabeçalhos para o arquivo CSV
    cabecalhos = ['Pesos Portfólio', 'Retorno', 'Volatilidade', 'Capital Final', 'IS']
    
    # Lista para armazenar os resultados de cada execução
    resultados = []
    
    # Executar o algoritmo 2 n vezes e armazenar os resultados
    for _ in range(n):
        resultado = resultados_tempera(precos)
        resultados.append({
            'Pesos Portfólio': ' '.join(map(lambda x: f"{x*100:.6f}%", resultado[0])),  # Converter a lista de pesos em string
            'Retorno': f"{resultado[1]*100:.6f}%",
            'Volatilidade': f"{resultado[2]*100:.6f}%",
            'Capital Final': f"{resultado[3]:.6f}",
            'IS': f"{resultado[4]:.6f}",
        })
    
    # Escrever os resultados em um arquivo CSV
    with open(nome_arquivo, 'w', newline='') as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=cabecalhos)
        escritor.writeheader()
        for resultado in resultados:
            escritor.writerow(resultado)
    
    print("CSV gerado.")
    
def vetor_para_df(vetor_pesos, num_ativos):
    # Determinando o número de execuções com base no tamanho do vetor e no número de ativos
    num_execucoes = len(vetor_pesos) // num_ativos
    
    # Verificando se o vetor pode ser dividido igualmente pelo número de ativos
    if len(vetor_pesos) % num_ativos != 0:
        raise ValueError("O tamanho do vetor não é divisível pelo número de ativos.")
    
    # Convertendo o vetor em uma matriz e depois para um DataFrame
    matriz_pesos = np.array(vetor_pesos).reshape(num_execucoes, num_ativos)
    df_pesos = pd.DataFrame(matriz_pesos, columns=lista_acoes)
    
    return df_pesos

def criar_mapa_calor_pesos(df_pesos, salvar_imagem=True):
    caminho_imagem=f'{folder}/mapa_calor_pesos_{ano_dos_ativos}'
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pesos.transpose(), cmap='viridis', annot=True, fmt=".2f")
    
    plt.title('Mapa de Calor dos Pesos do Portfólio')
    plt.xlabel('Execução')
    plt.ylabel('Ativo')
    
    if salvar_imagem:
        plt.savefig(f'{caminho_imagem}_{time.time()}.png', format='png', dpi=300)  # Salva a figura no caminh especificado com alta resolução  
   # plt.show()  # Mostra a figura

def plotar_metricas(df, salvar_imagem=True):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))  # Cria uma figura com subplots
    caminho_imagem = f'{folder}/grafico_{ano_dos_ativos}'

    # Temperatura x Iteração
    axs[0, 0].plot(df['Iteracao'], df['Temperatura'], color='blue')
    axs[0, 0].set_title('Temperatura x Iteração')
    axs[0, 0].set_xlabel('Iteração')
    axs[0, 0].set_ylabel('Temperatura')

    # IS x Iteração
    axs[0, 1].plot(df['Iteracao'], df['IS'], color='green')
    axs[0, 1].set_title('IS x Iteração')
    axs[0, 1].set_xlabel('Iteração')
    axs[0, 1].set_ylabel('IS')

    # Retorno x Iteração
    axs[1, 0].plot(df['Iteracao'], df['Retorno'], color='red')
    axs[1, 0].set_title('Retorno x Iteração')
    axs[1, 0].set_xlabel('Iteração')
    axs[1, 0].set_ylabel('Retorno')

    # Risco x Iteração
    axs[1, 1].plot(df['Iteracao'], df['Risco'], color='purple')
    axs[1, 1].set_title('Risco x Iteração')
    axs[1, 1].set_xlabel('Iteração')
    axs[1, 1].set_ylabel('Risco')

    plt.tight_layout()  # Ajusta automaticamente os parâmetros dos subplots para dar espaço entre eles
    if salvar_imagem:
        plt.savefig(f'{caminho_imagem}_{time.time()}.png', format='png', dpi=300)  # Salva a figura no caminh especificado com alta resolução
    # plt.show()

def resultados_tempera(precos):
    metricas_iteracoes = []
    variacao_precos = calcula_variacao_precos(precos)
    tot_dias_dados = len(variacao_precos)
    covariancia = calcula_covariancia_acoes(variacao_precos)
    taxas_de_retornos = calcula_retorno_periodo_total_acoes(precos)

    retorno_ativo_livre_risco = dict_retorno_ativo_livre_risco[ano_dos_ativos]
    pesos_portfolio = tempera_simulada(precos, taxas_de_retornos, covariancia, retorno_ativo_livre_risco, tot_dias_dados, metricas_iteracoes = [])
    
    df_metricas = pd.DataFrame(metricas_iteracoes)
    plotar_metricas(df_metricas)
    
    capital_inicial = 100000.0  # Cem mil reais
    
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxas_de_retornos)
    capital_final = capital_inicial * (1 + retorno_portfolio)
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tot_dias_dados)
    sharpe = (retorno_portfolio - retorno_ativo_livre_risco)/risco_portifolio
    
    df = vetor_para_df(pesos_portfolio, 6)
    criar_mapa_calor_pesos(df)
   
    return pesos_portfolio, retorno_portfolio, risco_portifolio, capital_final, sharpe

# Cada linha representa um valor de tempo (trimestre, semestre, etc), e cada coluna o valor da ação da empresa no fechamento
# precos = np.array([[100.0, 200.0, 300.0], [10.0, 210.0, 305.0], [50.0, 205.0, 310.0], [55.0, 208.0, 315.0]])

if (f'precos_{ano_dos_ativos}.csv' not in os.listdir()):
    lista_acoes_sa = [acao + ".SA" for acao in lista_acoes]
    inicio = dt.date(ano_dos_ativos, 1, 1)
    final = dt.date(ano_dos_ativos, 12, 31)
    precos = yf.download(lista_acoes_sa, inicio, final)['Close']
    
    # criar arquivo com os precos que foram baixados
    np.savetxt(f'precos_{ano_dos_ativos}.csv', precos, delimiter=",")
else:
    precos = np.loadtxt(f'precos_{ano_dos_ativos}.csv', delimiter=",")

gerar_csv(precos, f'resultados_ts_{ano_dos_ativos}.csv')