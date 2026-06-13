import tkinter as tk # importa a biblioteca para criar a janela
from tkinter import ttk, messagebox # importa componentes da janela (menus, popups)
import numpy as np # importa numpy para fazer cálculos com arrays
from Settings import numbers_set # importa os padrões dos números do ficheiro Settings
import time

def bipolarize(arr): # função que converte os valores binários (0 e 1) para bipolares (-1 e 1)
    return np.array(arr) * 2 - 1 # multiplica por 2 e subtrai 1: 0 vira -1, 1 vira 1


class Hebb: # classe do algoritmo de Hebb
    def __init__(self): # inicializa os pesos e o bias a None (ainda não treinados)
        self.W = None
        self.b = None

    def train(self, X, y_idx): # treina a rede com os padrões
        start_t = time.time()
        total_correct = 0
        total_error = 0

        n_numbers = len(set(y_idx)) # conta quantas classes existem (8 números)
        n_inputs = len(X[0]) # conta quantos pixels tem cada padrão (70)
        #cria um "quadro Branco"
        self.W = np.zeros((n_numbers, n_inputs)) # cria a matriz de pesos com zeros
        self.b = np.zeros(n_numbers) # cria o vetor de bias com zeros
        #Para cada desenho (xi) e para o seu número real correspondente (target), 
        #ele vai obrigar os 8 neurónios da saída a olhar para esse desenho.
        for xi, target in zip(X, y_idx): # para cada padrão e o seu número correto
            for i in range(n_numbers): # para cada classe (0 a 7)
                desired = 1 if i == target else -1 #+1 caso seja o número, -1 se for outro número
                self.W[i] += desired * xi # atualiza os pesos: regra de Hebb (W = W + g * f)
                self.b[i] += desired # atualiza o bias da mesma forma

        for xi, target in zip(X, y_idx):
            if self.predict(xi) == target:
                total_correct += 1
            else:
                total_error += 1


        final_t = time.time()
        accuracy = total_correct/len(X)
        return {
            "time":  final_t - start_t,
            "correct": total_correct,
            "wrong": total_error,
            "accuracy": accuracy,
            "epochs": "----",
            "iterations": "----",
            "mse": "----"
            }

    def predict(self, x): # classifica um padrão desenhado
        outputs = np.dot(self.W, x) + self.b # calcula a ativação de cada neurónio de saída
        return np.argmax(outputs) # devolve o índice do neurónio mais ativado (o número reconhecido)

#Pode falhar pq Se o teu número "1" partilhar muitos pixels parecidos com o
#teu número "7", a matriz de pesos vai ficar baralhada (haverá muita sobreposição de memória).

class Perceptron: # classe do algoritmo Perceptron
    def __init__(self, lr=0.2): # inicializa com learning rate 0.2 por defeito
        self.W = None
        self.b = None
        self.lr = lr # guarda o learning rate

    def train(self, X, y_idx, iterations=100): # treina durante 100 épocas (vezes)
        start_t = time.time()
        total_correct = 0
        total_error = 0
        n_numbers = len(set(y_idx)) # número de outputs
        n_inputs = len(X[0]) # número de inputs (pixels)
        self.W = np.zeros((n_numbers, n_inputs)) # pesos começam a zero
        self.b = np.zeros(n_numbers) # bias começa a zero
        epochs_run = 0
        for _ in range(iterations): # repete o treino durante o número de épocas definido
            epochs_run += 1
            has_error = False # check for early stopping
            for xi, target in zip(X, y_idx): # para cada padrão de treino
                for i in range(n_numbers): # para cada classe
                    out = np.sign(np.dot(self.W[i], xi) + self.b[i]) #y_in= np.dot(self.W[i], xi) + self.b[i], 
                                                                     #y_out = np.sign(y_in), (1(ativo) ou -1(inativo))
                    desired = 1 if i == target else -1 # valor esperado

                    error = desired - out # erro, caso a rede tenha acertado -1 no lugar de -1 ou 1 no lugar de 1, erro=0, 
                                          # caso contrário, Devia dar 1 e a rede disse -1: e=2. Devia dar -1 e a rede disse 1: e=-2.  
                    if error != 0:
                        self.W[i] += self.lr * error * xi # atualiza os pesos só se houver erro
                        self.b[i] += self.lr * error # atualiza o bias só se houver erro
                        has_error = True

            if not has_error:
                break # early stopping se não houver erros na época

        for xi, target in zip(X, y_idx):
            if self.predict(xi) == target:
                total_correct += 1
            else:
                total_error += 1

        final_t = time.time()
        accuracy = total_correct/len(X)
        return {
            "time":  final_t - start_t,
            "correct": total_correct,
            "wrong": total_error,
            "accuracy": accuracy,
            "epochs": "----",
            "iterations": epochs_run,
            "mse": "----"
            }
    
    def predict(self, x): # classifica um padrão
        outputs = np.dot(self.W, x) + self.b # calcula a ativação de cada neurónio
        return np.argmax(outputs) # devolve o número com maior ativação


class Adaline: # classe do algoritmo Adaline (regra delta)
    def __init__(self, lr=0.001): # learning rate pequeno para não divergir
        self.W = None
        self.b = None
        self.lr = lr

    def train(self, X, y_idx, epochs=200): # treina durante 200 épocas
        start_t = time.time()
        total_correct = 0
        total_error = 0
        n_numbers = len(set(y_idx))
        n_inputs = len(X[0])
        self.W = np.zeros((n_numbers, n_inputs)) # pesos a zero
        self.b = np.zeros(n_numbers) # bias a zero
        final_mse = 0
        for _ in range(epochs): # repete por todas as épocas
            W_gradient = np.zeros_like(self.W)
            b_gradient = np.zeros_like(self.b)
            epoch_mse = 0

            for xi, target in zip(X, y_idx): # para cada padrão
                desired = np.full(n_numbers, -1.0) # vetor com -1 em todas as classes
                desired[target] = 1.0 # coloca +1 na classe correta
                #cria uma lista com 8 posições cheia de -1. Depois, vai direto à posição do número correto (target) e muda-a para 1
                net = np.dot(self.W, xi) + self.b # calcula a saída linear (sem função sinal), guarda o numero exato. net=Y_in 
                error = desired - net # erro = valor esperado - valor calculado
                epoch_mse += np.sum(error ** 2)
                W_gradient += np.outer(error, xi) # acumula o erro para os pesos
                b_gradient += error # acumula o erro para o bias
            
            final_mse = epoch_mse / (len(X) * n_numbers)
            # atualiza os pesos uma vez por época (Batch Learning)
            self.W += self.lr * W_gradient
            self.b += self.lr * b_gradient
                

        for xi, target in zip(X, y_idx):
            if self.predict(xi) == target:
                total_correct += 1
            else:
                total_error += 1

        final_t = time.time()
        accuracy = total_correct/len(X)

        return {
        "time":  final_t - start_t,
        "correct": total_correct,
        "wrong": total_error,
        "accuracy": accuracy,
        "epochs": epochs,
        "iterations": "----",
        "mse": final_mse
        }

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
        self.model_choice_box = ttk.Combobox(controls, textvariable=self.model_var,
                     values=["Hebb", "Perceptron", "Adaline"],
                     state="readonly", width=18)
        self.model_choice_box.pack() # menu dropdown para escolher o algoritmo
        self.model_choice_box.bind("<<ComboboxSelected>>", self.update_ui_visibility)

        self.hyperparams_frame = tk.Frame(controls)
        self.hyperparams_frame.pack(fill="x")

        self.epochs_frame = tk.Frame(self.hyperparams_frame)
        tk.Label(self.epochs_frame, text="Epochs").pack(anchor="w")
        self.epochs = tk.StringVar(value="100")
        tk.Entry(self.epochs_frame, textvariable=self.epochs, width=10).pack()

        self.lr_frame = tk.Frame(self.hyperparams_frame)
        tk.Label(self.lr_frame, text="Learning Rate").pack(anchor="w") # etiqueta "Learning Rate"
        self.lr_var = tk.StringVar(value="0.2") # valor por defeito do learning rate
        tk.Entry(self.lr_frame, textvariable=self.lr_var, width=10).pack() # caixa de texto para o learning rate

        tk.Label(controls, text="Number Label").pack(anchor="w") # etiqueta para escolher o número a guardar
        self.save_label_var = tk.StringVar(value=self.y_labels[0]) # número selecionado por defeito
        ttk.Combobox(controls, textvariable=self.save_label_var,
                     values=self.y_labels, state="readonly", width=18).pack() # dropdown para escolher o número

        tk.Button(controls, text="Train Model", command=self.train_model).pack(fill="x", pady=2) # botão para treinar
        tk.Button(controls, text="Classify Drawing", command=self.classify_drawing).pack(fill="x", pady=2) # botão para classificar
        tk.Button(controls, text="Save as Example", command=self.save_example).pack(fill="x", pady=2) # botão para guardar exemplo
        tk.Button(controls, text="Clear Drawing", command=self.clear_drawing).pack(fill="x", pady=2) # botão para limpar a grelha

        tk.Button(root, text="📊 View Performance Dashboard", command=self.show_dashboard, bg="#d1ecf1").grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.result_label = tk.Label(controls, text="") # etiqueta que mostra o resultado
        self.result_label.pack()

        self.status_label = tk.Label(controls, text="", fg="gray") # etiqueta que mostra o estado do treino
        self.status_label.pack()

        self.performance_history = {
            "Hebb": {"time": "N/A", "correct": "N/A", "wrong": "N/A", "accuracy": "N/A", "epochs": "----", "iterations": "----", "mse": "----"},
            "Perceptron": {"time": "N/A", "correct": "N/A", "wrong": "N/A", "accuracy": "N/A", "epochs": "----", "iterations": "----", "mse": "----"},
            "Adaline": {"time": "N/A", "correct": "N/A", "wrong": "N/A", "accuracy": "N/A", "epochs": "----", "iterations": "----", "mse": "----"}
        }

        self.update_ui_visibility()

    def update_ui_visibility(self, event=None):
        model_choice = self.model_var.get()
        
        self.epochs_frame.pack_forget()
        self.lr_frame.pack_forget()
        
        if model_choice == "Adaline":
            self.epochs_frame.pack(fill="x")
            self.lr_frame.pack(fill="x")
            self.lr_var.set("0.001")
                
        elif model_choice == "Perceptron":
            self.lr_frame.pack(fill="x")
            self.lr_var.set("0.2")


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
        name = self.model_var.get() # lê o algoritmo selecionado

        lr = 0.2
        if name in ["Perceptron", "Adaline"]:
            try:
                lr = float(self.lr_var.get()) # lê o learning rate introduzido pelo utilizador
            except ValueError:
                messagebox.showerror("Invalid input", "Learning rate must be a number.") # erro se não for número
                return
        self.status_label.config(text=f"Training {name}...", fg="gray") # mostra que está a treinar
        self.root.update() # atualiza a janela para mostrar a mensagem

        if name == "Hebb": # se o algoritmo for Hebb
            self.hebb = Hebb() # cria um novo modelo Hebb
            statistics = self.hebb.train(self.X, self.y_idx) # treina com os padrões
            self.current_model = "Hebb" # guarda que o modelo atual é Hebb
            self.performance_history["Hebb"] = statistics

        elif name == "Perceptron": # se o algoritmo for Perceptron
            self.perceptron = Perceptron(lr) # cria um novo Perceptron com o learning rate dado
            statistics = self.perceptron.train(self.X, self.y_idx) # treina durante 100 épocas
            self.current_model = "Perceptron"
            self.performance_history["Perceptron"] = statistics

        else: # se o algoritmo for Adaline
            self.adaline = Adaline(lr) # cria um novo Adaline
            try:
                ep = int(self.epochs.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Epochs must be an integer.")
                self.status_label.config(text="Training failed.", fg="red")
                return
            statistics = self.adaline.train(self.X, self.y_idx, ep) # treina durante as épocas
            self.current_model = "Adaline"
            self.performance_history["Adaline"] = statistics

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


    def show_dashboard(self):
        # Create popup window
        dash_window = tk.Toplevel(self.root)
        dash_window.title("Algorithm Performance Evaluation")
        dash_window.geometry("800x300")
        
        tk.Label(dash_window, text="Algorithm Comparison Matrix", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Create Treeview table widget
        columns = ("metric", "hebb", "perceptron", "adaline")
        tree = ttk.Treeview(dash_window, columns=columns, show="headings", height=7)
        tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Define Headings
        tree.heading("metric", text="Evaluation Topic")
        tree.heading("hebb", text="Hebb")
        tree.heading("perceptron", text="Perceptron")
        tree.heading("adaline", text="Adaline (Widrow-Off)")
        
        # Adjust Column Widths
        tree.column("metric", width=180, anchor="center")
        tree.column("hebb", width=150, anchor="center")
        tree.column("perceptron", width=150, anchor="center")
        tree.column("adaline", width=180, anchor="center")
        
        # Pull latest historical data
        h_stats = self.performance_history["Hebb"]
        p_stats = self.performance_history["Perceptron"]
        w_stats = self.performance_history["Adaline"]
        
        # Formatting helper function
        def fmt(val, suffix="", precision=4):
            return f"{val:.{precision}f}{suffix}" if isinstance(val, (int, float)) else val

        # Insert rows for the 3 key topics
        tree.insert("", "end", values=("Training Time", fmt(h_stats['time'], " sec"), fmt(p_stats['time'], " sec"), fmt(w_stats['time'], " sec")))
        tree.insert("", "end", values=("Epochs", fmt(h_stats['epochs'], precision=0), fmt(p_stats['epochs'], precision=0), fmt(w_stats['epochs'], precision=0)))
        tree.insert("", "end", values=("Iterations", fmt(h_stats['iterations'], precision=0), fmt(p_stats['iterations'], precision=0), fmt(w_stats['iterations'], precision=0)))
        tree.insert("", "end", values=("Final MSE", fmt(h_stats['mse']), fmt(p_stats['mse']), fmt(w_stats['mse'])))
        tree.insert("", "end", values=("Correct", fmt(h_stats['correct'], precision=0), fmt(p_stats['correct'], precision=0), fmt(w_stats['correct'])))
        tree.insert("", "end", values=("Errors", fmt(h_stats['wrong'], precision=0), fmt(p_stats['wrong'], precision=0), fmt(w_stats['wrong'])))
        tree.insert("", "end", values=("Train Dataset Accuracy", fmt(h_stats['accuracy'], 1), fmt(p_stats['accuracy'], 1), fmt(w_stats['accuracy'], 1)))
        
        # Informational note footer
        tk.Label(dash_window, text="* Loss metric uses total classification errors for Perceptron and MSE for Adaline.", 
                 font=("Arial", 8, "italic"), fg="gray").pack(pady=5)


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
    
    
    root = tk.Tk() # cria a janela principal
    app = ANNApp(root) # inicia a aplicação
    root.mainloop() # mantém a janela aberta