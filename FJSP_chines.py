import pandas as pd 
import numpy as np 
import time
import pulp
import plotly.figure_factory as ff
import datetime
from operator import itemgetter

# declaramos o solver
solver = pulp.get_solver('GUROBI')

# resgatamos os dados
ordem_operacoes=pd.read_excel("Dados_DFJSP.xlsx", sheet_name="ordem_operacoes")
tempo_de_processamento=pd.read_excel("Dados_DFJSP.xlsx", sheet_name="tempo_de_processamento")

# transformamos os dados em matrizes
ops = ordem_operacoes.values
pt = tempo_de_processamento.values

# número de jobs
n_jobs = len(ordem_operacoes["Job"].unique())

# numero de operações
n_operations = len(tempo_de_processamento["Operacao"].unique())

# numero de job
n_maquinas = len(tempo_de_processamento["maquina"].unique())


# Declaração das variáveis do modelo
Cmax = pulp.LpVariable('Cmax',lowBound = 0, cat='Continuous')

Ci = pulp.LpVariable.dicts("start_time",
                           ((i) for i in range(1,n_jobs + 1)),
                           lowBound=0,
                           cat='Continuous')
#i=job,j=operation,k=maquina
st = pulp.LpVariable.dicts("start_time",
                           ((i, j, k) for i in range(1,n_jobs + 1) for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0] for k in list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina'])),
                           lowBound=0,
                           cat='Continuous')



ct = pulp.LpVariable.dicts("completed_time",
                           ((i, j, k) for i in range(1,n_jobs + 1) for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0] for k in list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina'])),
                           lowBound=0,
                           cat='Continuous')

x = pulp.LpVariable.dicts("X_binary_var",
                          ((i, j, k) for i in range(1,n_jobs + 1) for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0] for k in list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina'])),
                          lowBound=0, upBound=1,
                          cat='Binary')

y = pulp.LpVariable.dicts("Y_binary_var",
                          ((i, j, l, m,k) for i in range(1,n_jobs + 1) for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]  for l in range(1,6) for m in [ops[l-1][a] for a in range(1,5) if ops[l-1][a]!=0]  for k in list(set(list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina']))&set(list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==m),'maquina']))) if i < l ),
                          lowBound=0, upBound=1,
                          cat='Binary')


# declaração do modelo
model = pulp.LpProblem("MIN_makespan", pulp.LpMinimize)
model += Cmax


# Acrescentando as restrições ao modelo
for i in range(1,n_jobs + 1):
    for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]:
        ma_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina'])
        model += pulp.lpSum(x[i,j,ma] for ma in ma_list)==1


for i in range(1,n_jobs + 1):
    for j in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]:
        ma_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j),'maquina'])
        for k in ma_list:
            ptime=int(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==j)&(tempo_de_processamento['maquina']==k),[f"Job{i}"]].iloc[0])
            I = i
            J = j
            K = k
            model += st[I,J,K] + ct[I,J,K] <=x[I,J,K]*1000
            model += ct[I,J,K] >= st[I,J,K] +ptime-(1-x[I,J,K])*1000


for i in range(1,n_jobs + 1):
    for l in range(1,n_jobs + 1):
        if i<l:
            for o1 in [ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]:
                ma1_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==o1)&(tempo_de_processamento.loc[:][f"Job{i}"]!=0),'maquina'])
                for o2 in [ops[l-1][o] for o in range(1,n_operations - 1) if ops[l-1][o]!=0]:
                    ma2_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==o2)&(tempo_de_processamento.loc[:][f"Job{l}"]!=0),'maquina'])
                    mach_set=list(set(ma1_list)&set(ma2_list))
                    if len(mach_set)>0:
                        for k in mach_set:
                            model += st[i,o1,k]>=ct[l,o2,k]-y[i,o1,l,o2,k]*1000
                            model += st[l,o2,k]>=ct[i,o1,k]-(1-y[i,o1,l,o2,k])*1000


for i in range(1,n_jobs + 1):
    op_num=[ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]
    for j in range(1,len(op_num)):
        preop_num=op_num[j-1]
        curop_num=op_num[j]
        prema_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==preop_num)&(tempo_de_processamento.loc[:][f"Job{i}"]!=0),'maquina'])
        curma_list=list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==curop_num)&(tempo_de_processamento.loc[:][f"Job{i}"]!=0),'maquina'])
        model += pulp.lpSum(st[i,curop_num,ma1] for ma1 in curma_list) >= pulp.lpSum(ct[i,preop_num,ma2] for ma2 in prema_list)


for i in range(1,n_jobs + 1):
    op_list=[ops[i-1][o] for o in range(1,n_operations - 1) if ops[i-1][o]!=0]
    last_op=op_list[-1]
    model +=Ci[i]>=pulp.lpSum(ct[i,last_op,k] for k in list(tempo_de_processamento.loc[(tempo_de_processamento['Operacao']==last_op),'maquina']))
    model += Cmax >= Ci[i]

# resolvendo o modelo
model.solve(solver)
pulp.LpStatus[model.status]     

# resgarando os valores após as resolução
for var in ct:
    var_value = ct[var].varValue
    print( var[0],var[1],var[2] ,var_value)

total_cost = pulp.value(model.objective)
print ('min cost:',total_cost)

for i in x:
    print(i[0],i[1],i[2],'----',x[i].varValue)