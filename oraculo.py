# simuladorOraculo {
#       capitalInicial = 100.000;

#         vetorPreco01_01_2020 = [5, 8, 9, 3, 2]
#         vetorPreço31_12_2020 = [3, 10, 11, 2, 3]
#         vetorGanho = vetorPreço31_12_2020 - vetorPreco01_01_2020
#         # vetorGanho = [-2, 2, 2, -1, 1]
#         vetorGanhoRelativo = vetorGanho/vetorPreco01_01_2020
#         # vetorGanhoRelativo = [-2/5, 2/8, 2/8, -1/3, 1/2] = [-0,4, 0,25, 0.25, -0.33, 0.5]
#         somaP = somaPositivos(vetorGanhoRelativo) # 0.25 + 0.25 + 0.5 = 1.0
#         vetorInvest = substituiNegativosPorZero(vetorGanhoRelativo) # retorna [0, 0,25, 0.25, 0, 0.5]
#         vetorInvestNormalizado = dividePositivosPorSomaP(vetorGanho, somaP) # [0, 0.25/1, 0.25/1, 0, 0.5/1] = [0, 0,25, 0.25, 0, 0.5]

#         capitalInvestido = capitalInicial * vetorInvestNormalizado = 100.000 * [0, 0,25, 0.25, 0, 0.5] = [0, 25000, 25000, 0, 50000]

#         vetorCapitalFinal = capitalInvestido + capitalInvestido * vetorGanhoRelativo = [0, 25000, 25000, 0, 50000] + ([0, 25000, 25000, 0, 50000]* [-0,4, 0,25, 0.25, -0.33, 0.5] = [0, 31250, 31250, 0, 75000]

#         capitalFinal = somaValores([0, 31250, 31250, 0, 75000]) = 31250+31250+75000 = 137500
#         ganhoPercentual = (capitalFinal - capitalInicial)/capitalInicial # = (137500-100000)/100000 = 0,375 ou 37,5% de ganho total percentual

# }

# https://www.youtube.com/watch?v=BchQuTJvRAs

def somaPositivos(vetor):
    return sum(x for x in vetor if x > 0)

def substituiNegativosPorZero(vetor):
    return [max(0, x) for x in vetor]

def dividePositivosPorSomaP(vetor, somaP):
    return [0 if x <= 0 else x / somaP for x in vetor]

def somaValores(vetor):
    return sum(vetor)

def simuladorOraculo():
    capitalInicial = 100000

    vetorPreco01_01_2020 = [5, 8, 9, 3, 2]
    vetorPreco31_12_2020 = [3, 10, 11, 2, 3]
    vetorGanho = [p2 - p1 for p1, p2 in zip(vetorPreco01_01_2020, vetorPreco31_12_2020)]

    vetorGanhoRelativo = [(g / p1) if p1 > 0 else 0 for g, p1 in zip(vetorGanho, vetorPreco01_01_2020)]

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
