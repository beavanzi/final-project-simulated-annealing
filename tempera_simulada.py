import yfinance as yf
import numpy as np
import datetime as dt
import pandas as pd
import random
import math
import os
# def calcula_retorno_medio_acoes(dominio):
#     # calcular os retornos médios para cada ativo, axis=0 calcula a média ao longo das colunas
#     retornos_medios = np.mean(dominio, axis=0)
#     return retornos_medios

# def calcula_retorno_simples_acoes(dominio): # retorno esperado
#     # calcular os retornos simples para cada ativo, axis=0 calcula a média ao longo das colunas
#     matriz_retornos = np.array([(dominio[i] - dominio[i-1]) / dominio[i-1] for i in range(1, len(dominio))])
#     retorno = np.mean(matriz_retornos, axis=0)
#     return retorno

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

# def calcula_risco_diario_acoes(dominio):
#     # calcular o desvio padrão dos retornos para cada ativo, axis=0 calcula a média ao longo das colunas
#     risco_diario = np.std(dominio, axis=0)
#     return risco_diario

# def calcula_risco_periodo_total_acoes(dominio, risco_mensal):
#     # calcula o risco do período
#     n, m =  dominio.shape
#     risco_periodo = risco_mensal* np.sqrt(n)
#     return risco_periodo

# def sharpe_ratio_acoes(dominio):
#     retorno_ativo_livre_risco = 0.028364 
#     retorno = calcula_retorno_periodo_total_acoes(dominio)
#     risco_periodo = calcula_risco_periodo_total_acoes(dominio, calcula_risco_diario_acoes(dominio))
#     sharpe = (retorno - retorno_ativo_livre_risco)/risco_periodo
#     print("O sharpe é: ",   sharpe)
#     return sharpe

def calcula_covariancia_acoes(variacao_precos):
    # Wagner: a covariância estava errada, estava sendo calculada em relação à transposta dos preços, e o correto é em relação
    # à variação de preços (sem ser transposta)

    covariancia = variacao_precos.cov()
    return covariancia

def calcula_variacao_precos(precos):
    # convert numpy array para dataframe
    prices_df = pd.DataFrame(precos, columns=['Column_A', 'Column_B', 'Column_C'])
    # calcular a covariancia entre os ativos. cada elemento desta matriz representa a covariância (como variam) entre os retornos de dois ativos diferentes. a covariância é uma medida de como dois ativos se movimentam juntos.
    variacao_precos = prices_df / prices_df.shift(1) - 1
    # Retira os dados faltantes
    variacao_precos = variacao_precos.dropna()
    return variacao_precos



def calcula_volatilidade_portfolio(pesos_portfolio, covariancia, tamanho_amostra):
    # este é o produto de matrizes entre a matriz de covariância e o vetor de pesos do portfólio. o resultado é um novo vetor que representa a combinação linear da covariância de cada ativo, ponderada pelos pesos do portfólio.
    #combinacao_linear = np.dot(covariancia, portfolio)

    # fazemos outro produto de matrizes entre o vetor transposto de pesos do portfólio e o vetor de combinacao linear.  o resultado é um único valor numérico que representa a variância total do portfólio.
    #variancia = np.dot(portfolio.T, combinacao_linear)

    # a raiz quadrada da variância é o desvio padrão, que é o que normalmente definimos como volatilidade. portanto, o resultado final da expressão é a volatilidade do portfólio.
    #volatilidade = np.sqrt(variancia)

    #return volatilidade

    desvio_padrao_por_dia = np.sqrt(np.dot(np.dot(pesos_portfolio.T, covariancia), pesos_portfolio))

    desvio_padrao_total = desvio_padrao_por_dia * np.sqrt(tamanho_amostra)  # 1 ano ou 2
    return desvio_padrao_total


def funcao_objetivo(pesos_portfolio, precos, taxa_retorno, retornoAtivoLivreRisco):
    qut_dias_variacoes_preco = len(precos)-1

    # acho que está errado retorno = calcula_retorno_portfolio(pesos_portfolio, taxas_de_retornos)
    risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia,qut_dias_variacoes_preco)
    # print("w O risco é: ",risco)
    retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno) # wagn errado: taxas_de_retornos * pesos_portfolio
    # print("W As taxas_de_retorno são: ", taxas_de_retornos,". O pesos do portfolio são: ", pesos_portfolio)
    # print("w O retorno do portfolio é: ",np.sum(retorno_portfolio))
    # wagn erro: sharpe = np.sum(retorno_portfolio) / risco  # Agora estamos maximizando
    sharpe = (retorno_portfolio - retornoAtivoLivreRisco) / risco_portifolio
    #print("w o sharpe é: ",sharpe,'. o retorno do potifolio: ',retorno_portfolio,". o risco: ", risco_portifolio, ".\n", "Os pesos do portfolio são: ",pesos_portfolio)

    return sharpe


def probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura):
    # print("wagn melhor valor objetivo: ",melhor_valor_objetivo, ". novo_valor_objetivo: ",novo_valor_objetivo)
    # sorteio = np.random.rand()
    # probabilidade = np.exp((melhor_valor_objetivo - novo_valor_objetivo) / temperatura) # rever esse calculo 
    
    sorteio = random.random()
    probabilidade = pow(math.e, (- novo_valor_objetivo - melhor_valor_objetivo) / temperatura)
    #print("sorteio: ",sorteio, " probabilidade: ", probabilidade)
    return  sorteio < probabilidade


def normaliza_vetor(vetor):
    normalizado = []
    for i in range(len(vetor)):
        normalizado.append((vetor[i] - min(vetor)) / (max(vetor) - min(vetor)));
    return normalizado / np.sum(normalizado)

def distribui_vetor(vetor):
    return vetor / np.sum(vetor)

def tempera_simulada(precos, taxa_retorno, retornoAtivoLivreRisco, temperatura=1000.0, resfriamento=0.9999, passo=0.05):
    n, m = precos.shape
    pesos_portfolio = gerar_primeira_solucao(precos)

    pesos_melhor_portfolio = np.copy(pesos_portfolio)
    melhor_valor_objetivo = funcao_objetivo(pesos_melhor_portfolio, precos, taxa_retorno, retornoAtivoLivreRisco)

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
        novo_valor_objetivo = funcao_objetivo(novo_portfolio, precos, taxa_retorno, retornoAtivoLivreRisco)

        #print("wag novo valor objetivo: ",novo_valor_objetivo, "melhor valor anterior: ",melhor_valor_objetivo)
        #print("\n")

        prob_aceita_solucao_ruim = probabilidade_aceitacao(melhor_valor_objetivo, novo_valor_objetivo, temperatura)

        if (novo_valor_objetivo > melhor_valor_objetivo or prob_aceita_solucao_ruim):
            #print("Aceitou solução ruim: ", prob_aceita_solucao_ruim)
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

def ajustar_soma_para_um(numeros):
    """
    Ajusta o último número da lista para garantir que a soma total seja 1,
    mantendo os valores como números de ponto flutuante.

    Parâmetros:
    numeros (list of float): Lista de números em ponto flutuante.

    Retorna:
    list of float: Lista de números ajustados.
    """
    soma_atual = sum(numeros)
    
    # Calculando a diferença necessária para ajustar a soma para 1
    diferenca = 1 - soma_atual
    
    # Ajustando o último número
    numeros[-1] += diferenca
    
    # Certificando-se de que todos os números estejam arredondados para até 6 casas decimais
    numeros_ajustados = np.array([round(numero, 6) for numero in numeros])
    
    return numeros_ajustados

def formatar_e_ajustar_numeros(numeros):
    """
    Recebe um array de números em notação científica,
    remove a notação científica, arredonda para até 6 casas decimais,
    e ajusta a soma para ser exatamente 1, mantendo os números como float.

    Parâmetros:
    numeros (list): Lista de números em notação científica.

    Retorna:
    list: Lista de números ajustados em ponto flutuante.
    """
    # Arredondando os números para até 6 casas decimais
    numeros_arredondados = np.array([round(numero, 6) for numero in numeros])
    
    # Ajustando os números para garantir que a soma seja 1
    numeros_ajustados = ajustar_soma_para_um(numeros_arredondados)
    
    return numeros_ajustados


#Cada linha representa um valor de tempo (trimestre, semestre, etc), e cada coluna o valor da ação da empresa no fechamento
precos = np.array([[100.0, 200.0, 300.0], [10.0, 210.0, 305.0], [50.0, 205.0, 310.0], [55.0, 208.0, 315.0]])

# if ("precos.csv" not in os.listdir()):
#     lista_acoes = ["VALE3", "PETR3", "ITUB3", "BBDC3", "JBSS3", "BBAS3"]
#     lista_acoes = [acao + ".SA" for acao in lista_acoes]
#     inicio = dt.date(2023, 1, 1)
#     final = dt.date(2023, 12, 31)
#     precos = yf.download(lista_acoes, inicio, final)['Close']
#     # criar arquivo com os precos que foram baixados
#     np.savetxt("precos.csv", precos, delimiter=",")
# else:
#     precos = np.loadtxt("precos.csv", delimiter=",")


# wagn: a covariância deve ser calculada apenas 1 vez, pois ela sempre é a mesma
variacao_precos = calcula_variacao_precos(precos)
tot_dias_dados = len(variacao_precos)
covariancia = calcula_covariancia_acoes(variacao_precos)
taxas_de_retornos = calcula_retorno_periodo_total_acoes(precos)
# ativo livre de risco (selic para 3 meses) precisa ser o mesmo período de dados
retornoAtivoLivreRisco = 0.028364
# pesos_portfolio = np.array([0.01, 0.8, 0.19])
pesos_portfolio = tempera_simulada(precos, taxas_de_retornos, retornoAtivoLivreRisco)
pesos_portfolio = formatar_e_ajustar_numeros(pesos_portfolio)
# print('Solucao inicial: ', gerar_primeira_solucao(precos))
print('Pesos portfolio: ', pesos_portfolio)

capital_inicial = 100000.0  # Cem mil reais

taxa_retorno = calcula_retorno_periodo_total_acoes(precos)
print("WA taxa de retorno é: ", taxa_retorno)
retorno_portfolio = calcula_retorno_portfolio(pesos_portfolio, taxa_retorno)
print('Retorno portfolio: ', retorno_portfolio)
capital_final = capital_inicial * (1 + retorno_portfolio)
print("Capital final: ", capital_final)
#print("Volatilidade:", calcula_volatilidade_portfolio(pesos_portfolio, covariancia,tot_dias_dados))


wDf = pd.DataFrame(pesos_portfolio)
t = pd.DataFrame(taxa_retorno) * wDf

risco_portifolio = calcula_volatilidade_portfolio(pesos_portfolio, covariancia,tot_dias_dados)
print("w o risco (volatilidade) do portifolio é: ",risco_portifolio)
retorno_portifolio_fict = calcula_retorno_portfolio(np.array([0.01, 0.8, 0.19]), taxa_retorno)
print("w o retorno do portfolio ficticio é: ", retorno_portifolio_fict)
print("\n")

sharpe = (retorno_portfolio-retornoAtivoLivreRisco)/risco_portifolio
#funcao_objetivo(pesos_portfolio,precos)
print("w o sharpe é: ",sharpe)
