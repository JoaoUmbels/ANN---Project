import tkinter as tk # importa a biblioteca para criar a janela
from tkinter import ttk, messagebox # importa componentes da janela (menus, popups)
import numpy as np # importa numpy para fazer cálculos com arrays
from Settings import numbers_set # importa os padrões dos números do ficheiro Settings


def bipolarize(arr): # função que converte os valores binários (0 e 1) para bipolares (-1 e 1)
    return np.array(arr) * 2 - 1 # multiplica por 2 e subtrai 1: 0 vira -1, 1 vira 1


class Hebb: # classe do algoritmo de Hebb
    def __init__(self): # inicializa os pesos e o bias a None (ainda não treinados)
        self.W = None
        self.b = None

    def train(self, X, y_idx): # treina a rede com os padrões
        n_classes = len(set(y_idx)) # conta quantas classes existem (8 números)
        n_inputs = len(X[0]) # conta quantos pixels tem cada padrão (70)
        self.W = np.zeros((n_classes, n_inputs)) # cria a matriz de pesos com zeros
        self.b = np.zeros(n_classes) # cria o vetor de bias com zeros
        for xi, target in zip(X, y_idx): # para cada padrão e o seu número correto
            for i in range(n_classes): # para cada classe (0 a 7)
                desired = 1 if i == target else -1 # valor esperado: +1 se for a classe certa, -1 se não
                self.W[i] += desired * xi # atualiza os pesos: regra de Hebb (W = W + g * f)
                self.b[i] += desired # atualiza o bias da mesma forma

    def predict(self, x): # classifica um padrão desenhado
        outputs = np.dot(self.W, x) + self.b # calcula a ativação de cada neurónio de saída
        return np.argmax(outputs) # devolve o índice do neurónio mais ativado (o número reconhecido)


class Perceptron: # classe do algoritmo Perceptron
    def __init__(self, lr=0.2): # inicializa com learning rate 0.2 por defeito
        self.W = None
        self.b = None
        self.lr = lr # guarda o learning rate

    def train(self, X, y_idx, epochs=100): # treina durante 100 épocas
        n_classes = len(set(y_idx)) # número de classes
        n_inputs = len(X[0]) # número de inputs (pixels)
        self.W = np.zeros((n_classes, n_inputs)) # pesos começam a zero
        self.b = np.zeros(n_classes) # bias começa a zero
        for _ in range(epochs): # repete o treino durante o número de épocas definido
            for xi, target in zip(X, y_idx): # para cada padrão de treino
                for i in range(n_classes): # para cada classe
                    out = np.sign(np.dot(self.W[i], xi) + self.b[i]) # calcula a saída com função sinal (+1 ou -1)
                    desired = 1 if i == target else -1 # valor esperado
                    error = desired - out # calcula o erro
                    self.W[i] += self.lr * error * xi # atualiza os pesos só se houver erro
                    self.b[i] += self.lr * error # atualiza o bias só se houver erro

    def predict(self, x): # classifica um padrão
        outputs = np.dot(self.W, x) + self.b # calcula a ativação de cada neurónio
        return np.argmax(outputs) # devolve o número com maior ativação


class Adaline: # classe do algoritmo Adaline (regra delta)
    def __init__(self, lr=0.001): # learning rate pequeno para não divergir
        self.W = None
        self.b = None
        self.lr = lr

    def train(self, X, y_idx, epochs=200): # treina durante 200 épocas
        n_classes = len(set(y_idx))
        n_inputs = len(X[0])
        self.W = np.zeros((n_classes, n_inputs)) # pesos a zero
        self.b = np.zeros(n_classes) # bias a zero
        for _ in range(epochs): # repete por todas as épocas
            for xi, target in zip(X, y_idx): # para cada padrão
                desired = np.full(n_classes, -1.0) # vetor com -1 em todas as classes
                desired[target] = 1.0 # coloca +1 na classe correta
                net = np.dot(self.W, xi) + self.b # calcula a saída linear (sem função sinal)
                error = desired - net # erro = valor esperado - valor calculado
                self.W += self.lr * np.outer(error, xi) # atualiza pesos com a regra delta
                self.b += self.lr * error # atualiza bias

    def predict(self, x): # classifica um padrão
        outputs = np.dot(self.W, x) + self.b # calcula a ativação linear
        return np.argmax(outputs) # devolve o número mais provável


class ANNApp: # classe principal que cria a interface gráfica
    def __init__(self, root): # inicializa a aplicação
        self.root = root
        self.root.title("ANN Numbers Classifier") # título da janela

        self.rows = 10 # grelha tem 10 linhas
        self.cols = 7 # grelha tem 7 colunas
        self.cell_size = 50 # cada célula tem 50 pixels de lado

        self.draws = {k: bipolarize(v).flatten() for k, v in numbers_set.items()} # converte os padrões para bipolar e achata para vetor
        self.X = list(self.draws.values()) # lista com os vetores de treino
        self.y_labels = list(self.draws.keys()) # lista com os nomes dos números ("0" a "7")
        self.y_idx = list(range(len(self.y_labels))) # lista com os índices (0 a 7)

        self.hebb = None # modelo Hebb ainda não treinado
        self.perceptron = None # modelo Perceptron ainda não treinado
        self.adaline = None # modelo Adaline ainda não treinado
        self.current_model = None # guarda qual o modelo atualmente selecionado

        self.grid_state = np.zeros((self.rows, self.cols), dtype=int) # grelha começa toda a zeros (branco)

        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size,
                                height=self.rows * self.cell_size, bg="white") # cria a área de desenho
        self.canvas.grid(row=0, column=1, padx=10, pady=10) # coloca o canvas na janela
        self.canvas.bind("<Button-1>", self.on_click) # liga o clique do rato à função on_click

        self.rects = [] # lista para guardar todos os retângulos da grelha
        for r in range(self.rows): # para cada linha
            row_rects = []
            for c in range(self.cols): # para cada coluna
                x1, y1 = c * self.cell_size, r * self.cell_size # canto superior esquerdo
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size # canto inferior direito
                rect = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                    fill="white", outline="black") # desenha o retângulo
                row_rects.append(rect) # guarda o retângulo na lista
            self.rects.append(row_rects) # guarda a linha de retângulos

        self.create_controls() # cria os botões e menus do lado esquerdo

    def create_controls(self): # função que cria os controlos da interface
        controls = tk.Frame(self.root, padx=10, pady=10) # cria um painel para os controlos
        controls.grid(row=0, column=0, sticky="n") # coloca o painel à esquerda

        tk.Label(controls, text="Algorithm").pack(anchor="w") # etiqueta "Algorithm"
        self.model_var = tk.StringVar(value="Perceptron") # variável que guarda o algoritmo selecionado
        ttk.Combobox(controls, textvariable=self.model_var,
                     values=["Hebb", "Perceptron", "Adaline"],
                     state="readonly", width=18).pack() # menu dropdown para escolher o algoritmo

        tk.Label(controls, text="Learning Rate").pack(anchor="w") # etiqueta "Learning Rate"
        self.lr_var = tk.StringVar(value="0.2") # valor por defeito do learning rate
        tk.Entry(controls, textvariable=self.lr_var, width=10).pack() # caixa de texto para o learning rate

        tk.Label(controls, text="Number Label").pack(anchor="w") # etiqueta para escolher o número a guardar
        self.save_label_var = tk.StringVar(value=self.y_labels[0]) # número selecionado por defeito
        ttk.Combobox(controls, textvariable=self.save_label_var,
                     values=self.y_labels, state="readonly", width=18).pack() # dropdown para escolher o número

        tk.Button(controls, text="Train Model", command=self.train_model).pack(fill="x", pady=2) # botão para treinar
        tk.Button(controls, text="Classify Drawing", command=self.classify_drawing).pack(fill="x", pady=2) # botão para classificar
        tk.Button(controls, text="Save as Example", command=self.save_example).pack(fill="x", pady=2) # botão para guardar exemplo
        tk.Button(controls, text="Clear Drawing", command=self.clear_drawing).pack(fill="x", pady=2) # botão para limpar a grelha

        self.result_label = tk.Label(controls, text="") # etiqueta que mostra o resultado
        self.result_label.pack()

        self.status_label = tk.Label(controls, text="", fg="gray") # etiqueta que mostra o estado do treino
        self.status_label.pack()

    def on_click(self, event): # função chamada quando o utilizador clica na grelha
        col = event.x // self.cell_size # calcula em que coluna foi o clique
        row = event.y // self.cell_size # calcula em que linha foi o clique
        if 0 <= row < self.rows and 0 <= col < self.cols: # verifica se o clique está dentro da grelha
            self.grid_state[row, col] = 1 - self.grid_state[row, col] # inverte o estado da célula (0 vira 1 e 1 vira 0)
            color = "black" if self.grid_state[row, col] else "white" # escolhe a cor
            self.canvas.itemconfig(self.rects[row][col], fill=color) # pinta a célula

    def clear_drawing(self): # função para limpar a grelha
        self.grid_state.fill(0) # coloca todos os valores a zero
        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.itemconfig(self.rects[r][c], fill="white") # pinta todas as células de branco
        self.result_label.config(text="") # limpa o resultado
        self.status_label.config(text="Drawing cleared.") # mostra mensagem

    def train_model(self): # função chamada quando se clica em "Train Model"
        try:
            lr = float(self.lr_var.get()) # lê o learning rate introduzido pelo utilizador
        except ValueError:
            messagebox.showerror("Invalid input", "Learning rate must be a number.") # erro se não for número
            return

        name = self.model_var.get() # lê o algoritmo selecionado
        self.status_label.config(text=f"Training {name}...", fg="gray") # mostra que está a treinar
        self.root.update() # atualiza a janela para mostrar a mensagem

        if name == "Hebb": # se o algoritmo for Hebb
            self.hebb = Hebb() # cria um novo modelo Hebb
            self.hebb.train(self.X, self.y_idx) # treina com os padrões
            self.current_model = "Hebb" # guarda que o modelo atual é Hebb

        elif name == "Perceptron": # se o algoritmo for Perceptron
            self.perceptron = Perceptron(lr) # cria um novo Perceptron com o learning rate dado
            self.perceptron.train(self.X, self.y_idx, epochs=100) # treina durante 100 épocas
            self.current_model = "Perceptron"

        else: # se o algoritmo for Adaline
            self.adaline = Adaline(lr) # cria um novo Adaline
            self.adaline.train(self.X, self.y_idx, epochs=200) # treina durante 200 épocas
            self.current_model = "Adaline"

        self.status_label.config(text=f"{name} training completed!", fg="black") # mostra que o treino acabou

    def classify_drawing(self): # função chamada quando se clica em "Classify Drawing"
        if self.current_model is None: # verifica se já foi treinado algum modelo
            messagebox.showwarning("No model", "Please train a model first.")
            return

        x = bipolarize(self.grid_state).flatten() # converte a grelha para vetor bipolar

        if self.current_model == "Hebb": # usa o modelo correto para classificar
            pred = self.hebb.predict(x)
        elif self.current_model == "Perceptron":
            pred = self.perceptron.predict(x)
        else:
            pred = self.adaline.predict(x)

        self.result_label.config(text=f"Recognized: {self.y_labels[pred]}") # mostra o número reconhecido

    def save_example(self): # função para guardar o desenho atual como exemplo de treino
        label = self.save_label_var.get() # lê o número escolhido
        x = bipolarize(self.grid_state).flatten() # converte a grelha para vetor
        self.X.append(x) # adiciona o novo exemplo à lista de treino
        self.y_labels.append(label) # adiciona a etiqueta
        self.y_idx.append(self.y_labels.index(label)) # adiciona o índice correspondente
        messagebox.showinfo("Saved", f"Example saved as '{label}'.") # mostra mensagem de confirmação


if __name__ == "__main__":
    
    print("""
══════════════════════════════════════════════════════════════
                ANN NUMBERS' CLASSIFIER
══════════════════════════════════════════════════════════════

Supported Numbers
──────────────────────
0, 1, 2, 3, 4, 5, 6, 7

Learning Algorithms
──────────────────────
- Adaline
- Hebb
- Perceptron

Instructions
─────────────────────────────────────────────────────────────
1. Open the drawing canvas.
2. Select and train an algorithm.
3. Draw one of the supported numbers.
4. Click "Classify Drawing" to get a prediction.
5. Select a number and click "Save as Example".
6. Use "Clear" to erase the canvas and try again.
══════════════════════════════════════════════════════════════
""")
    input("Press ENTER to open the drawing window...")
    launch_gui()
    root = tk.Tk() # cria a janela principal
    app = ANNApp(root) # inicia a aplicação
    root.mainloop() # mantém a janela aberta