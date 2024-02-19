import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd
import random
import math
import os

def calcula_retorno_portfolio(portfolio, retornos):
    # calcula a soma ponderada dos retornos dos ativos, ponderada pelos respectivos pesos no portfólio, e retorna esse valor como o retorno total esperado do portfólio
    return np.sum(portfolio * retornos)

def calcula_retorno_periodo_total_acoes(dominio):
    # calcular os retornos simples para cada ativo, ultimo valor da coluna - primeiro valor da coluna / primeiro valor da coluna
    n, m = dominio.shape
    array_retornos = np.zeros(m)
    for j in range(m):
        array_retornos[j] = (dominio[-1][j] - dominio[0][j]) / dominio[0][j]
    return array_retornos

def calcula_covariancia_acoes(variacao_precos):
    # Wagner: a covariância estava errada, estava sendo calculada em relação à transposta dos preços, e o correto é em relação
    # à variação de preços (sem ser transposta)
    covariancia = variacao_precos.cov()
    return covariancia

def calcula_variacao_precos(precos):
    # convert numpy array para dataframe
    prices_df = pd.DataFrame(precos, columns=['Column_A', 'Column_B', 'Column_C', 'Column_D', 'Column_E', 'Column_F'])
    # calcular a covariancia entre os ativos. cada elemento desta matriz representa a covariância (como variam) entre os retornos de dois ativos diferentes. a covariância é uma medida de como dois ativos se movimentam juntos.
    variacao_precos = prices_df / prices_df.shift(1) - 1
    # Retira os dados faltantes
    variacao_precos = variacao_precos.dropna()
    return variacao_precos

def calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tamanho_amostra):
    
    desvio_padrao_por_dia = np.sqrt(np.dot(np.dot(pesos_portfolio.T, covariancia), pesos_portfolio))
    desvio_padrao_total = desvio_padrao_por_dia * np.sqrt(tamanho_amostra)
    
    return desvio_padrao_total


def funcao_objetivo(pesos_portfolio, precos, taxa_retorno, covariancia, retorno_ativo_livre_risco):
    qut_dias_variacoes_preco = len(precos)-1
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, qut_dias_variacoes_preco)
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno) # wagn errado: taxas_de_retornos * pesos_portfolio
    sharpe = (retorno_portfolio - retorno_ativo_livre_risco) / risco_portifolio
    #print("w o sharpe é: ",sharpe,'. o retorno do potifolio: ',retorno_portfolio,". o risco: ", risco_portifolio, ".\n", "Os pesos do portfolio são: ",pesos_portfolio)

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

def tempera_simulada(precos, taxa_retorno, covariancia, retorno_ativo_livre_risco, temperatura=1000.0, resfriamento=0.9999, passo=0.05):
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
        contador += 1

        temperatura = temperatura * resfriamento
    print('Contador: ', contador)
    return pesos_melhor_portfolio


def gerar_primeira_solucao(precos):
    # para diminuir a aleatoriedade: calcular os pesos de retorno do dominio normalizados ou usar tudo igual
    n, m = precos.shape
    unitario = [1] * m
    solucao = distribui_vetor(unitario)

    # calcular os retornos para cada ativo
    # retornos = calcula_retorno_periodo_total_acoes(precos)
    # solucao = normaliza_vetor(retornos)
    print("Solucao inicial: ", solucao)
    return solucao

# def gerar_csv(nome_arquivo):
#     n = 10
    
#     # Cabeçalhos para o arquivo CSV
#     cabecalhos = ['Pesos Portfólio', 'Retorno', 'Volatilidade', 'Capital Final', 'IS']
    
#     # Lista para armazenar os resultados de cada execução
#     resultados = []
    
#     # Executar o algoritmo 2 n vezes e armazenar os resultados
#     for _ in range(n):
#         resultado = algoritmo_2()
#         resultados.append({
#             'Pesos Portfólio': ' '.join(map(str, resultado[0])),  # Converter a lista de pesos em string
#             'Retorno': resultado[1],
#             'Volatilidade': resultado[2],
#             'Capital Final': resultado[3],
#             'IS': resultado[4],
#         })
    
#     # Escrever os resultados em um arquivo CSV
#     with open(nome_arquivo, 'w', newline='') as arquivo_csv:
#         escritor = csv.DictWriter(arquivo_csv, fieldnames=cabecalhos)
#         escritor.writeheader()
#         for resultado in resultados:
#             escritor.writerow(resultado)


# Cada linha representa um valor de tempo (trimestre, semestre, etc), e cada coluna o valor da ação da empresa no fechamento
# precos = np.array([[100.0, 200.0, 300.0], [10.0, 210.0, 305.0], [50.0, 205.0, 310.0], [55.0, 208.0, 315.0]])

if ("precos.csv" not in os.listdir()):
    lista_acoes = ["VALE3", "PETR3", "ITUB3", "BBDC3", "JBSS3", "BBAS3"]
    lista_acoes = [acao + ".SA" for acao in lista_acoes]
    inicio = dt.date(2023, 1, 1)
    final = dt.date(2023, 12, 31)
    precos = yf.download(lista_acoes, inicio, final)['Close']
    
    # criar arquivo com os precos que foram baixados
    np.savetxt("precos.csv", precos, delimiter=",")
else:
    precos = np.loadtxt("precos.csv", delimiter=",")


# wagn: a covariância deve ser calculada apenas 1 vez, pois ela sempre é a mesma
variacao_precos = calcula_variacao_precos(precos)
tot_dias_dados = len(variacao_precos)
covariancia = calcula_covariancia_acoes(variacao_precos)
taxas_de_retornos = calcula_retorno_periodo_total_acoes(precos)

# ativo livre de risco (selic para o ano de 2023), precisa ser o mesmo período de dados
retorno_ativo_livre_risco = 0.1225
pesos_portfolio = tempera_simulada(precos, taxas_de_retornos, covariancia, retorno_ativo_livre_risco)
print('Pesos portfolio: ', pesos_portfolio)

capital_inicial = 100000.0  # Cem mil reais

taxa_retorno = calcula_retorno_periodo_total_acoes(precos)
print("WA taxa de retorno é: ", taxa_retorno)
retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno)
print('Retorno portfolio: ', retorno_portfolio)
capital_final = capital_inicial * (1 + retorno_portfolio)
print("Capital final: ", capital_final)
#print("Volatilidade:", calcula_volatilidade_portfolio(pesos_portfolio, covariancia,tot_dias_dados))


# wDf = pd.DataFrame(pesos_portfolio)
# t = pd.DataFrame(taxa_retorno) * wDf

risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tot_dias_dados)
print("w o risco (volatilidade) do portifolio é: ", risco_portifolio)
# retorno_portifolio_fict = calcula_retorno_portfolio(np.array([0.01, 0.8, 0.19]), taxa_retorno)
# print("w o retorno do portfolio ficticio é: ", retorno_portifolio_fict)
# print("\n")

sharpe = (retorno_portfolio-retorno_ativo_livre_risco)/risco_portifolio
print("w o sharpe é: ", sharpe)
