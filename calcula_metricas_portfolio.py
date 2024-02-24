import numpy as np
import pandas as pd

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
    variacao_precos = prices_df / prices_df.shift(1) - 1
    variacao_precos = variacao_precos.dropna()
    return variacao_precos

def calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tamanho_amostra):
    
    desvio_padrao_por_dia = np.sqrt(np.dot(np.dot(pesos_portfolio.T, covariancia), pesos_portfolio))
    desvio_padrao_total = desvio_padrao_por_dia * np.sqrt(tamanho_amostra)
    
    return desvio_padrao_total

def calcula_metricas(pesos_portfolio, precos, capital_inicial):
    variacao_precos = calcula_variacao_precos(precos)
    tot_dias_dados = len(variacao_precos)
    covariancia = calcula_covariancia_acoes(variacao_precos)
    taxas_de_retornos = calcula_retorno_periodo_total_acoes(precos)
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxas_de_retornos)
    capital_final = capital_inicial * (1 + retorno_portfolio)
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tot_dias_dados)
    
    return pesos_portfolio, retorno_portfolio, risco_portifolio, capital_final
    
ano_dos_ativos = 2019
capital_inicial = 100000.0
pesos_raw = [0.000000, 0.001154, 0.002441, 17.490722, 51.378214, 31.127468]
pesos_portfolio = np.array([peso / 100 for peso in pesos_raw])

precos = np.loadtxt(f'precos_{ano_dos_ativos}.csv', delimiter=",")

result = calcula_metricas(pesos_portfolio, precos, capital_inicial)
print(f'Retorno: {result[1]*100}%. Volatilidade: {result[2]*100}%. Capital Final: R$ {result[3]}.')