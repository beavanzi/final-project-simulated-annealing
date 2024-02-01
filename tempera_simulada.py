import yfinance as yf
import numpy as np
import datetime as dt
import os

def calcula_retorno_simples_acoes(dominio):
    # calcular os retornos simples para cada ativo, axis=0 calcula a média ao longo das colunas
    matriz_retornos = np.array([(dominio[i] - dominio[i-1]) / dominio[i-1] for i in range(1, len(dominio))])
    retorno = np.mean(matriz_retornos, axis=0)
    return retorno

# def calcula_retorno_medio_acoes(dominio):
#     # calcular os retornos médios para cada ativo, axis=0 calcula a média ao longo das colunas
#     retornos_medios = np.mean(dominio, axis=0)
#     return retornos_medios

def calcula_covariancia_acoes(dominio):
    # calcular a covariancia entre os ativos. cada elemento desta matriz representa a covariância (como variam) entre os retornos de dois ativos diferentes. a covariância é uma medida de como dois ativos se movimentam juntos.
    covariancia = np.cov(dominio.T)
    return covariancia

def calcula_retorno_portfolio(portfolio, retornos):
    # calcula a soma ponderada dos retornos dos ativos, ponderada pelos respectivos pesos no portfólio, e retorna esse valor como o retorno total esperado do portfólio
    return np.sum(portfolio * retornos)

def calcula_volatilidade_portfolio(portfolio, covariancia):
    # este é o produto de matrizes entre a matriz de covariância e o vetor de pesos do portfólio. o resultado é um novo vetor que representa a combinação linear da covariância de cada ativo, ponderada pelos pesos do portfólio.
    combinacao_linear = np.dot(covariancia, portfolio)
    
    # fazemos outro produto de matrizes entre o vetor transposto de pesos do portfólio e o vetor de combinacao linear.  o resultado é um único valor numérico que representa a variância total do portfólio.
    variancia = np.dot(portfolio.T, combinacao_linear)

    # a raiz quadrada da variância é o desvio padrão, que é o que normalmente definimos como volatilidade. portanto, o resultado final da expressão é a volatilidade do portfólio.
    volatilidade = np.sqrt(variancia)
    return volatilidade

def funcao_objetivo(portfolio, dominio):
    retornos = calcula_retorno_simples_acoes(dominio)
    # print('Retornos:', retornos)
    # print('Dominio:', dominio)
    covariancia = calcula_covariancia_acoes(dominio)
    # print('Covariancia:', covariancia)
    
    retorno = calcula_retorno_portfolio(portfolio, retornos)
    risco = calcula_volatilidade_portfolio(portfolio, covariancia)
    sharpe = retorno / risco  # Agora estamos maximizando
    return -sharpe  # Negativo porque estamos usando um otimizador de minimização

def probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
    return np.random.rand() < np.exp((melhor_valor_objetivo - novo_valor_objetivo) / temperatura)

# função para normalizar um vetor para que a soma seja igual a 1
def normaliza_vetor(vetor):
    return vetor / np.sum(vetor)

def tempera_simulada(dominio, temperatura = 1000000.0, resfriamento = 0.99, passo = 0.05):
    n, m = dominio.shape
    portfolio_atual = gerar_primeira_solucao(dominio) 

    melhor_portfolio = np.copy(portfolio_atual)
    melhor_valor_objetivo = funcao_objetivo(melhor_portfolio, dominio)
    contador = 0

    # enquanto a temperatura não foi quase zerada
    while temperatura > 0.1:
        novo_portfolio = np.copy(portfolio_atual)
        
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
        contador += 1
        
        temperatura = temperatura * resfriamento
    print('Contador: ', contador)
    return melhor_portfolio

def gerar_primeira_solucao(dominio):
    # para diminuir a aleatoriedade: calcular os pesos de retorno do dominio normalizados ou usar tudo igual
    # n, m = dominio.shape
    # unitario = [1] * m
    # solucao = normaliza_vetor(unitario)
    
    # calcular os retornos para cada ativo
    retornos = calcula_retorno_simples_acoes(dominio)
    solucao = normaliza_vetor(retornos)
    return solucao

# Cada linha representa um valor de tempo (trimestre, semestre, etc), e cada coluna o valor da ação da empresa no fechamento
precos = np.array([[100.0, 200.0, 300.0], [10.0, 210.0, 305.0], [50.0, 205.0, 310.0], [55.0, 208.0, 315.0]])

# if ("precos.csv" not in os.listdir()):
#     lista_acoes = ["VALE3", "PETR3", "ITUB3", "BBDC3", "JBSS3", "BBAS3"]
#     lista_acoes = [acao + ".SA" for acao in lista_acoes]
#     inicio = dt.date(2019, 1, 1)
#     final = dt.date(2022, 12, 31)
#     precos = yf.download(lista_acoes, inicio, final)['Close']
#     # criar arquivo com os precos que foram baixados
#     np.savetxt("precos.csv", precos, delimiter=",")
# else:
#     precos = np.loadtxt("precos.csv", delimiter=",")

pesos_portfolio = tempera_simulada(precos)
print('Pesos portfolio: ', pesos_portfolio)

capital_inicial = 100000.0          # Cem mil reais

retorno = calcula_retorno_simples_acoes(precos)
retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, retorno)
print('Retorno acoes: ', retorno)
print('Retorno portfolio: ', retorno_portfolio)
capital_final = capital_inicial * (1 + retorno_portfolio) 
print("Capital final: ", capital_final)
print("Volatilidade:", calcula_volatilidade_portfolio(pesos_portfolio, calcula_covariancia_acoes(precos)))