# %% [markdown]
# # Projeto Programming - Regressão Linear

# %% [markdown]
# Regressão Linear Simples : f_wb = w * x + b
# 
# Regressão Linear Múltipla: f_wb = $\overrightarrow{w}$ . $\overrightarrow{x}$ + b

# %% [markdown]
# ## Criação do dataset

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def generate_dataset(n_points, dimensions = 2, mean = 5, std_dev = 0.25):
    """Vai dar retorno a um dataset de vertical stack vetores com (n_points) data points 
    e (dimensions) colunas"""

    points = np.random.normal(loc= mean, scale= std_dev, size= (n_points, dimensions))        

    return np.vstack(points)

# %%
# Exemplo de criação de um dataset com 150 pontos e 2 colunas (1 coluna para a feature e outra para a label)

dataset = generate_dataset(150, dimensions = 2)

dataset

# %% [markdown]
# ## Iniciação da Regressão

# %%
def initial_lin_reg(x, y):

    b = np.random.uniform(1, 3)

    if x.ndim == 1:                                         # Se definirmos apenas 1 feature

        w = np.random.uniform(0, 1)                         # Gerar coeficiente de w de 0 ou 1

        y_hats = w * x + b                                  # Fazer previsões de y dado o w e b definidos acima

        plt.scatter(x, y)
        plt.plot(x, y_hats, color = 'red', label='Regressão Inicial')
        print(f'A representação inicial de y_hat = {w} x + {b}')

    else:                                                   # Se definirmos mais do que 1 feature no modelo

        w = np.random.randint(0,1,x.shape[1])               # Gerar coeficientes de w de 0 a 1 para cada feature

        y_hats = np.dot(x, w) + b

        formula_regressao = " + ".join([f"{w[i]}*x{i+1}" for i in range(len(w))]) if len(w) > 1 else f"{w[0]}x + {b}"
        print(f'A representação inicial de y_hat = {formula_regressao} + {b}')

    return w, b

# %%
# Exemplo para testar a função initial_lin_reg


# x, y, w, b = initial_lin_reg(dataset[:, :-1], dataset[:, -1])           # Caso tenhamos + do que 1 feature

w, b = initial_lin_reg(dataset[:,0], dataset[:,1])                # Caso tenhamos apenas 1 feature

# %% [markdown]
# ## Definição da função de custo - Mean Squared Error Function

# %%
def cost_function(x, y, w, b):

    m = len(x)                                              # Número de data points

    if x.ndim == 1:
        
        y_hats = w * x + b                                  # Previsões de y caso a regressão seja simples
    
    else:

        y_hats = np.dot(x, w) + b                           # Previsões de y caso a regressão seja múltipla    

    cost = (1/(2*m)) * np.sum((y_hats - y) ** 2)            # Função custo

    return y_hats, cost

# %%
y_hats, cost = cost_function(dataset[:,0], dataset[:,1], w, b)
y_hats, cost

# %% [markdown]
# # Regressão Linear

# %%
def linear_regression(x, y, learning_rate = 0.01, n_iterations = 100000, min_error = 1e-6, min_update = 1e-6):
    """ Esta função recria o processo de regressão linear simples/múltipla. Inicia parâmetros aleatórios de w e b, ficando
    assim com uma linha de referência. Depois, através do conceito de cost function e gradiente descendente, o algoritmo vai atualizar
    os parâmetros de w e b de forma a otimizar o modelo de regressão, até ao ponto em que w e b convergem (ou até à condição de paragem 
    definida pelo utilizador ser atingida: número de passos, erro mínimo ou alterações entre iterações mínima);
    
    É devolvido o valor final de w, o valor final de b, e o histórico da função custo;

    x = features/input
    y = label/ouput
    learning_rate = tamanho da atualização dos parâmetros no processo de gradiente descendente
    n_iterations = número máximo de atualizações de w e b
    min_error = threshold de cost function para o algoritmo parar
    min_update = threshold de atualizações em w e b para o algoritmo parar
    
    """

    w, b = initial_lin_reg(x, y)
    cost_history = []
    m = len(x)

    for i in range(n_iterations):

        y_hats, cost = cost_function(x, y, w, b)

        if x.ndim == 1:
            dcdw = (1/m) * np.sum((y_hats - y) * x)         # Derivada da cost function em relação a w
        else: 
            dcdw = (1/m) * np.dot(x.T, (y_hats - y))        # Derivada da cost function em relação a w quando temos mais do que 1 variável indep.
            
        dcdb = (1/m) * np.sum(y_hats - y)                   # Derivada da cost function em relação a b

        w_new = w - learning_rate * dcdw                    # Atualização de w
        b_new = b - learning_rate * dcdb                    # Atualização de b

        current_cost = cost_function(x, y, w_new, b_new)[1]

        cost_history.append(current_cost)

        if (np.array_equal(w_new, w) if not np.isscalar(w) else w_new == w) and b_new == b:                                 # Se não houver mais atualizações dos parâmetros
                print(f'Paragem na iteração {i + 1} porque a os valores de w e b convergiram de acordo com o gradiente.')
                break

        if current_cost < min_error:
                print(f'Paragem na iteração {i + 1} porque a função custo atual ({current_cost}) já não é superior a {min_error}.')
                break

        if np.all(np.abs(w_new - w) < min_update) and np.abs(b_new - b) < min_update:
                print(f'Paragem na iteração {i + 1} porque as alterações em w e b já não são maiores do que {min_update}.')
                break
        
        w, b = w_new, b_new
        
    else:
        print(f'Paragem porque atingiu o número máximo de iterações: {i + 1}')

    
    if x.ndim == 1:
        print(f'O ótimo da regressão está definido em y_hat = {w} x + {b}')
        plt.scatter(x, y)
        plt.plot(x, w* x + b, color = 'green', label='Regressão Final', linewidth=2)
        plt.legend()
        plt.title('Representação do dataset e regressões lineares')

    else:
        formula_regressao = " + ".join([f"{w[i]}*x{i+1}" for i in range(len(w))]) if len(w) > 1 else f"{w[0]}x + {b}"
        print(f'O ótimo da regressão está definido em y_hat = {formula_regressao} + {b}')
    
    return w, b, cost_history

# %%
linear_regression(dataset[:,0], dataset[:,1], learning_rate = 0.01, n_iterations=50000)

# %% [markdown]
# ## Aplicação prática da regressão

# %%
horas_estudo = np.array([8,0.5,2,3.4,9,13,10,3.7,8,7.7,14,11,4.8,3.7,12,1.75,6,9,8,15,5,6,7,8,9,10,11,12,13,14,15,0.75,1.75,2.4,3.9,5.84,6.95,5,8.75,9,11.5,13,16,12.85,13,14,10.5,13,12.75,10])

nota_exame = np.array([14,5,8,7.5,12,16,14,10,11,13,18,16,7.5,9,19,4.75,3.5,14,12.5,12,6,4.75,12,14,10,16.5,18,17,15.5,16,18,4.5,7,5,8,3.75,9,11,9.5,7.6,14.5,16,18.5,17.75,14.5,19,17,15.5,16,13])

# %%
horas_estudo.shape, nota_exame.shape

# %%
w, b, cost_history = linear_regression(horas_estudo,nota_exame, learning_rate = 0.01, n_iterations = 20000, min_error = 1e-12, min_update = 1e-12)

# %%
def predict_y(X, w, b):
    """ Para prever outputs tendo a(s) feature(s) definidas e os parâmetros de w e b otimizados"""

    if isinstance(X, (int, float)):                             # Verifica se X é apenas 1 feature 
        result = w * X + b
    else:                                                       # Caso X seja uma lista (assim como w)
        result = sum(x * wi for x, wi in zip(X, w)) + b

    return round(result,2)

# %%
resultado = predict_y(14,w,b)
resultado


