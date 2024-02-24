def somaPositivos(vetor):
    return sum(x for x in vetor if x > 0)

def substituiNegativosPorZero(vetor):
    return [max(0, x) for x in vetor]

def dividePositivosPorSomaP(vetor, somaP):
    return [0 if x <= 0 else x / somaP for x in vetor]

def somaValores(vetor):
    return sum(vetor)

def simuladorOraculo():
    capitalInicial = 100000 # Cem mil reais

    #vetorPreco01_01_2020 = [5, 8, 9, 3, 2]
    vetorPrecoInicio = [1.614999961853027344e+01,2.528800392150878906e+01,3.700000000000000000e+01,2.405999946594238281e+01,4.391283035278320312e+01,5.109000015258789062e+01]
    vetorPrecoFinal = [1.867000007629394531e+01,2.717505645751953125e+01,3.709999847412109375e+01,3.018000030517578125e+01,4.733061981201171875e+01,5.329999923706054688e+01]
    #vetorPreco31_12_2020 = [3, 10, 11, 2, 3]
    vetorGanho = [p2 - p1 for p1, p2 in zip(vetorPrecoInicio, vetorPrecoFinal)]

    vetorGanhoRelativo = [(g / p1) if p1 > 0 else 0 for g, p1 in zip(vetorGanho, vetorPrecoInicio)]

    somaP = somaPositivos(vetorGanhoRelativo)

    vetorInvest = substituiNegativosPorZero(vetorGanhoRelativo)

    vetorInvestNormalizado = dividePositivosPorSomaP(vetorInvest, somaP)

    capitalInvestido = [capitalInicial * x for x in vetorInvestNormalizado]

    vetorCapitalFinal = [inv + inv * g for inv, g in zip(capitalInvestido, vetorGanhoRelativo)]

    capitalFinal = somaValores(vetorCapitalFinal)

    ganhoPercentual = (capitalFinal - capitalInicial) / capitalInicial

    return capitalFinal, ganhoPercentual


# Exemplo de utilização do simuladorOraculo
capitalFinal, ganhoPercentual = simuladorOraculo()
print(f"Capital Final: {capitalFinal}")
print(f"Ganho Percentual: {ganhoPercentual * 100}%")



# Entender o que o oraculo tem a ver com o algoritmo de otminização de carteira de Markowitz
