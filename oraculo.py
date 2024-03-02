# def somaPositivos(vetor):
#     return sum(x for x in vetor if x > 0)

# def substituiNegativosPorZero(vetor):
#     return [max(0, x) for x in vetor]

# def dividePositivosPorSomaP(vetor, somaP):
#     return [0 if x <= 0 else x / somaP for x in vetor]

# def somaValores(vetor):
#     return sum(vetor)

# def simuladorOraculo():
#     capitalInicial = 100000 # Cem mil reais

#     # Dados de 2019 para os 6 ativos do portfólio
#     vetorPrecoInicio = [1.614999961853027344e+01,2.528800392150878906e+01,3.700000000000000000e+01,2.405999946594238281e+01,4.391283035278320312e+01,5.109000015258789062e+01]
#     vetorPrecoFinal = [1.867000007629394531e+01,2.717505645751953125e+01,3.709999847412109375e+01,3.018000030517578125e+01,4.733061981201171875e+01,5.329999923706054688e+01]
    
#     vetorGanho = [p2 - p1 for p1, p2 in zip(vetorPrecoInicio, vetorPrecoFinal)]
#     vetorGanhoRelativo = [(g / p1) if p1 > 0 else 0 for g, p1 in zip(vetorGanho, vetorPrecoInicio)]
#     somaP = somaPositivos(vetorGanhoRelativo)
#     vetorInvest = substituiNegativosPorZero(vetorGanhoRelativo)
#     vetorInvestNormalizado = dividePositivosPorSomaP(vetorInvest, somaP)
#     capitalInvestido = [capitalInicial * x for x in vetorInvestNormalizado]
#     vetorCapitalFinal = [inv + inv * g for inv, g in zip(capitalInvestido, vetorGanhoRelativo)]
#     capitalFinal = somaValores(vetorCapitalFinal)
#     ganhoPercentual = (capitalFinal - capitalInicial) / capitalInicial

#     return capitalFinal, ganhoPercentual

# capitalFinal, ganhoPercentual = simuladorOraculo()
# print(f"Capital Final: {capitalFinal}")
# print(f"Ganho Percentual: {ganhoPercentual * 100}%")

def soma_positivos(vetor):
    return sum(x for x in vetor if x > 0)

def substitui_negativos_por_zero(vetor):
    return [max(0, x) for x in vetor]

def divide_positivos_por_soma_p(vetor, soma_p):
    return [0 if x <= 0 else x / soma_p for x in vetor]

def soma_valores(vetor):
    return sum(vetor)

def simulador_oraculo():
    capital_inicial = 100000 # Cem mil reais

    # Dados de 2019 para os 6 ativos do portfólio
    vetor_preco_inicio = [1.614999961853027344e+01, 2.528800392150878906e+01, 3.700000000000000000e+01, 2.405999946594238281e+01, 4.391283035278320312e+01, 5.109000015258789062e+01]
    vetor_preco_final = [1.867000007629394531e+01, 2.717505645751953125e+01, 3.709999847412109375e+01, 3.018000030517578125e+01, 4.733061981201171875e+01, 5.329999923706054688e+01]
    
    vetor_ganho = [p2 - p1 for p1, p2 in zip(vetor_preco_inicio, vetor_preco_final)]
    vetor_ganho_relativo = [(g / p1) if p1 > 0 else 0 for g, p1 in zip(vetor_ganho, vetor_preco_inicio)]
    soma_p = soma_positivos(vetor_ganho_relativo)
    vetor_invest = substitui_negativos_por_zero(vetor_ganho_relativo)
    vetor_invest_normalizado = divide_positivos_por_soma_p(vetor_invest, soma_p)
    capital_investido = [capital_inicial * x for x in vetor_invest_normalizado]
    vetor_capital_final = [inv + inv * g for inv, g in zip(capital_investido, vetor_ganho_relativo)]
    capital_final = soma_valores(vetor_capital_final)
    ganho_percentual = (capital_final - capital_inicial) / capital_inicial

    return capital_final, ganho_percentual

capital_final, ganho_percentual = simulador_oraculo()
print(f"Capital Final: {capital_final}")
print(f"Ganho Percentual: {ganho_percentual * 100}%")
