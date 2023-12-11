import numpy as np
# import datetime as dt
# import yfinance as yf

# Função para normalizar um vetor para que a soma seja igual a 1
def normaliza_vetor(vetor):
    return vetor / np.sum(vetor)

# Função para calcular a volatilidade de um portfólio
def calcula_volatilidade(portfolio, covariancia):
    desvio_padrao = np.sqrt(np.dot(np.dot(portfolio.T, covariancia), portfolio))
    return desvio_padrao

# Função para calcular o retorno de um portfólio
def calcula_retorno(portfolio, retornos):
    return np.dot(portfolio, retornos.T)

# Função objetivo - nesse caso, maximizar o índice de Sharpe (retorno/volatilidade)
def funcao_objetivo(portfolio, retornos, covariancia, peso_retorno=0.8):
    retorno = calcula_retorno(portfolio, retornos)
    risco = calcula_volatilidade(portfolio, covariancia)
    sharpe = retorno / risco  # Agora estamos maximizando
    return -sharpe  # Negativo porque estamos usando um otimizador de minimização

def probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
    return np.random.rand() < np.exp((melhor_valor_objetivo - novo_valor_objetivo) / temperatura)

# Simulated Annealing
def simulated_annealing(retornos, covariancia, max_carteiras=100000, temperatura_inicial=1.0, fator_resfriamento=0.95, debug=False):
    n, m = retornos.shape
    portfolio_atual = normaliza_vetor(np.random.rand(m))  # Ajuste para usar m ao invés de n

    melhor_portfolio = np.copy(portfolio_atual)
    melhor_valor_objetivo = funcao_objetivo(melhor_portfolio, retornos, covariancia)

    temperatura = temperatura_inicial

    for _ in range(max_carteiras):
        novo_portfolio = np.copy(portfolio_atual)
        # Gera dois índices aleatórios e troca os valores entre eles
        i, j = np.random.choice(range(m), size=2, replace=False)  # Ajuste para usar m ao invés de n
        novo_portfolio[i], novo_portfolio[j] = novo_portfolio[j], novo_portfolio[i]
        novo_portfolio = normaliza_vetor(novo_portfolio)

        novo_valor_objetivo = funcao_objetivo(novo_portfolio, retornos, covariancia)
        if debug:
            print("novo_valor_objetivo: ", novo_valor_objetivo)

        # Aceita o novo ponto com probabilidade
        if all(novo_valor_objetivo > melhor_valor_objetivo) or all(probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura)):
            portfolio_atual = np.copy(novo_portfolio)
            melhor_portfolio = np.copy(portfolio_atual)
            melhor_valor_objetivo = novo_valor_objetivo

        temperatura *= fator_resfriamento
        if debug:
            print("T: ", temperatura, " Melhor resultado: ", melhor_valor_objetivo, " Resultado atual: ", novo_valor_objetivo)

    return melhor_portfolio

# Exemplo controlado de utilização
retornos = np.array([[1, 4, 3, 6],
                     [2, 2, 3, 6],
                     [5, 4, 2, 6],
                     [1, 5, 1, 6],
                     [7, 4, 2, 5]])
covariancia = np.cov(retornos, rowvar=False)

# Series Históricas Reais
# inicio = dt.date(2015, 1, 1)
# final = dt.date(2015, 12, 31)

# lista_acoes = ["WEGE3", "LREN3", "VALE3", "PETR3", "EQTL3", "EGIE3"]
# lista_acoes = [acao + ".SA" for acao in lista_acoes]

# precos = yf.download(lista_acoes, inicio, final)['Adj Close']
# matriz_retornos = precos.pct_change().apply(lambda x: np.log(1+x)).dropna() #retorno logaritmo

# retornos = np.array(matriz_retornos.values) #retorno logaritmo
# media_retornos = retornos.mean()
# matriz_cov = matriz_retornos.cov()
# covariancia = np.array(matriz_cov.values)

capital_inicial = 100000
pesos_portfolio = simulated_annealing(retornos, covariancia, debug=True)
vetor_retorno = calcula_retorno(pesos_portfolio, retornos)

print("Pesos do Portfolio otimizado:", pesos_portfolio)
print("Capital final: ", np.sum(capital_inicial * vetor_retorno))
print("Volatilidade:", calcula_volatilidade(pesos_portfolio, covariancia))