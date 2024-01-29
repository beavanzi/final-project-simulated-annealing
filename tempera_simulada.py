import yfinance as yf
import numpy as np
import datetime as dt

def funcao_custo():
    return 0

def calcula_retorno_portfolio(portfolio, retornos):
    return np.sum(portfolio * retornos)

def calcula_volatilidade_portfolio(portfolio, covariancia):
    volatilidade = np.sqrt(np.dot(portfolio.T, np.dot(covariancia, portfolio))) #rever a questao teorica desse caluclo
    return volatilidade

def funcao_objetivo(portfolio, dominio):
    retornos = np.mean(dominio, axis=0) # retorno médio
    covariancia = np.cov(dominio.T) # covariancia
    
    retorno = calcula_retorno_portfolio(portfolio, retornos)
    risco = calcula_volatilidade_portfolio(portfolio, covariancia)
    sharpe = retorno / risco  # Agora estamos maximizando
    return -sharpe  # Negativo porque estamos usando um otimizador de minimização

def probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
    return np.random.rand() < np.exp((melhor_valor_objetivo - novo_valor_objetivo) / temperatura)

# Função para normalizar um vetor para que a soma seja igual a 1
def normaliza_vetor(vetor):
    return vetor / np.sum(vetor)

def tempera_simulada(dominio, temperatura = 1000000.0, resfriamento = 0.99, passo = 0.05):
    n, m = dominio.shape
    portfolio_atual = gerar_primeira_solucao(dominio) 

    melhor_portfolio = np.copy(portfolio_atual)
    melhor_valor_objetivo = funcao_objetivo(melhor_portfolio, dominio)
    contador = 0

    # Enquanto a temperatura não foi quase zerada
    while temperatura > 0.1:
        novo_portfolio = np.copy(portfolio_atual)
        
        # Escolhendo valores e trocando as posições deles. Está certo?
        # i, j = np.random.choice(range(m), size=2, replace=False)
        # novo_portfolio[i], novo_portfolio[j] = novo_portfolio[j], novo_portfolio[i]
        # escolher uma posiçao e alterar ela de acordo com o passo.
        i = np.random.choice(range(m), size=1, replace=False)
        # direcao é um valor float aleatorio entre -passo e +passo
        direcao = np.random.uniform(-passo, passo)

        
        if novo_portfolio[i] + direcao > 0:
            novo_portfolio[i] = novo_portfolio[i] + direcao
        else:
            novo_portfolio[i] = novo_portfolio[i] - direcao
        
        novo_portfolio = normaliza_vetor(novo_portfolio)
        novo_valor_objetivo = funcao_objetivo(novo_portfolio, dominio)
        
        if (novo_valor_objetivo > melhor_valor_objetivo) or probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
            portfolio_atual = np.copy(novo_portfolio)
            melhor_portfolio = np.copy(portfolio_atual)
            melhor_valor_objetivo = novo_valor_objetivo
        #print(melhor_valor_objetivo)
        contador += 1
        #print(contador)
        
        temperatura = temperatura * resfriamento
    print(contador)
    return melhor_portfolio

def gerar_primeira_solucao(dominio):
    # para diminuir a aleatoriedade: calcular os pesos de retorno do dominio normalizados ou usar tudo igual
    n, m = dominio.shape
    unitario = [1] * m
    solucao = normaliza_vetor(unitario)
    return solucao

# Cada linha representa um valor de tempo (trimestre, semestre, etc), e cada coluna o valor da ação da empresa no fechamento
# historico = np.array([[100.0, 200.0, 300.0], [10.0, 210.0, 305.0], [50.0, 205.0, 310.0], [55.0, 208.0, 315.0]])

lista_acoes = ["VALE3", "PETR3", "ITUB3", "BBDC3", "JBSS3", "BBAS3"]
lista_acoes = [acao + ".SA" for acao in lista_acoes]
inicio = dt.date(2019, 1, 1)
final = dt.date(2022, 12, 31)
precos = yf.download(lista_acoes, inicio, final)['Adj Close']

pesos_portfolio = tempera_simulada(precos)
print(pesos_portfolio)

capital_inicial = 100000.0          # Cem mil reais
print("Capital final: ", np.sum(capital_inicial * calcula_retorno_portfolio(pesos_portfolio, retornos = np.mean(precos, axis=0) )))
print("Volatilidade:", calcula_volatilidade_portfolio(pesos_portfolio, covariancia = np.cov(precos.T)))