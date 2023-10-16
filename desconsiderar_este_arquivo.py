import pandas as pd
import matplotlib.pyplot as plt

# Dados
data = {
    "Questão": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "Robô - Fortemente Concordo": [76.9, 69.23, 84.62, 69.23, 57.69],
    "Robô - Concordo": [23.07, 26.92, 11.54, 19.23, 34.62],
    "Robô - Neutro": [0, 3.85, 0, 11.54, 7.69],
    "Convencional - Fortemente Concordo": [73.07, 76.92, 84.62, 57.69, 61.54],
    "Convencional - Concordo": [26.92, 23.07, 0, 34.62, 34.62],
    "Convencional - Neutro": [0, 0, 0, 3.85, 0]
}

df = pd.DataFrame(data)

# Plot
bar_width = 0.35
index = range(len(df["Questão"]))

fig, ax = plt.subplots(figsize=(12, 8))

bar1 = ax.bar(index, df["Robô - Fortemente Concordo"], bar_width, label="Robô - Fortemente Concordo", color='b')
bar2 = ax.bar(index, df["Robô - Concordo"], bar_width, label="Robô - Concordo", color='c', bottom=df["Robô - Fortemente Concordo"])
bar3 = ax.bar(index, df["Robô - Neutro"], bar_width, label="Robô - Neutro", color='r', bottom=[i+j for i,j in zip(df["Robô - Fortemente Concordo"], df["Robô - Concordo"])])

bar4 = ax.bar([i+bar_width for i in index], df["Convencional - Fortemente Concordo"], bar_width, label="Convencional - Fortemente Concordo", color='g')
bar5 = ax.bar([i+bar_width for i in index], df["Convencional - Concordo"], bar_width, label="Convencional - Concordo", color='y', bottom=df["Convencional - Fortemente Concordo"])
bar6 = ax.bar([i+bar_width for i in index], df["Convencional - Neutro"], bar_width, label="Convencional - Neutro", color='m', bottom=[i+j for i,j in zip(df["Convencional - Fortemente Concordo"], df["Convencional - Concordo"])])

ax.set_xlabel('Questão')
ax.set_ylabel('Porcentagem (%)')
ax.set_title('Comparação entre Comunicação via Robô e Método Convencional')
ax.set_xticks([i+bar_width/2 for i in index])
ax.set_xticklabels(df["Questão"])
ax.legend(loc="upper left", bbox_to_anchor=(1,1))

plt.tight_layout()
plt.show()
