import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from Settings import numbers_set

def bipolarize(arr):
    return np.array(arr) * 2 - 1


class Perceptron:
    def __init__(self, input_dim, n_classes, lr=0.2):
        self.W = np.zeros((n_classes, input_dim))
        self.b = np.zeros(n_classes)
        self.lr = lr

    def train(self, X, y_idx, epochs=50):
        for _ in range(epochs):
            for xi, target in zip(X, y_idx):
                for i in range(len(self.b)):
                    out = np.sign(np.dot(self.W[i], xi) + self.b[i])
                    desired = 1 if i == target else -1
                    error = desired - out
                    self.W[i] += self.lr * error * xi
                    self.b[i] += self.lr * error

    def predict(self, x):
        outputs = np.dot(self.W, x) + self.b
        return np.argmax(outputs)


class Adaline:
    def __init__(self, input_dim, n_classes, lr=0.01):
        self.W = np.zeros((n_classes, input_dim))
        self.b = np.zeros(n_classes)
        self.lr = lr

    def train(self, X, y_idx, epochs=100):
        for _ in range(epochs):
            for xi, target in zip(X, y_idx):
                desired = -np.ones(len(self.b))
                desired[target] = 1
                net = np.dot(self.W, xi) + self.b
                error = desired - net
                self.W += self.lr * np.outer(error, xi)
                self.b += self.lr * error

    def predict(self, x):
        outputs = np.dot(self.W, x) + self.b
        return np.argmax(outputs)


class ANNapp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Numbers' Classifier")

        self.rows, self.cols = 10, 7
        self.cell_size = 50

        self.num_draws = numbers_set.copy()
        self.draws = {k: bipolarize(v).flatten() for k, v in self.num_draws.items()}
        self.X = list(self.draws.values())
        self.y_labels = list(self.draws.keys())
        self.y_idx = list(range(len(self.y_labels)))

        self.perceptron = None
        self.adaline = None
        self.current_model_name = None
        # ADICIONAR AQUI UM DPS P O HEBB

        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg="white")
        self.canvas.grid(row=0, column=1, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.grid_state = np.zeros((self.rows, self.cols), dtype=int)

        self.rects = []
        for r in range(self.rows):
            row_rects = []
            for c in range(self.cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                row_rects.append(rect)
            self.rects.append(row_rects)

        self.create_controls()

    def create_controls(self):
        controls = tk.Frame(self.root, padx=10, pady=10)
        controls.grid(row=0, column=0, sticky="n")

        tk.Label(controls, text="Algorithm").pack(anchor="w")
        self.model_var = tk.StringVar(value="Perceptron")
        ttk.Combobox(controls, textvariable=self.model_var, values=["Perceptron", "Adaline"], state="readonly", width=18).pack()

        tk.Label(controls, text="Learning Rate").pack(anchor="w")
        self.lr_var = tk.StringVar(value="0.2")
        tk.Entry(controls, textvariable=self.lr_var, width=10).pack()

        tk.Label(controls, text="Number Label").pack(anchor="w")
        self.save_label_var = tk.StringVar(value=self.y_labels[0])
        ttk.Combobox(controls, textvariable=self.save_label_var, values=self.y_labels, state="readonly", width=18).pack()

        tk.Button(controls, text="Train Model", command=self.train_model).pack(fill="x", pady=2)
        tk.Button(controls, text="Classify Drawing", command=self.classify_drawing).pack(fill="x", pady=2)
        tk.Button(controls, text="Save as Example", command=self.save_example).pack(fill="x", pady=2)
        tk.Button(controls, text="Clear Drawing", command=self.clear_drawing).pack(fill="x", pady=2)

        self.status_label = tk.Label(controls, text="Train a model to start.", fg="blue", wraplength=150)
        self.status_label.pack(pady=5)

        self.result_label = tk.Label(controls, text="")
        self.result_label.pack()

    def on_canvas_click(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid_state[row, col] = 1 - self.grid_state[row, col]
            color = "black" if self.grid_state[row, col] else "white"
            self.canvas.itemconfig(self.rects[row][col], fill=color)

    def clear_drawing(self):
        self.grid_state.fill(0)
        for r in range(self.rows):
            for c in range(self.cols):
                self.canvas.itemconfig(self.rects[r][c], fill="white")
        self.result_label.config(text="")
        self.status_label.config(text="Drawing cleared.")

    def train_model(self):
        try:
            lr = float(self.lr_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Learning rate must be a number.")
            return

        name = self.model_var.get()
        self.status_label.config(text=f"Training {name}...", fg="gray")
        self.root.update()

        input_dim = len(self.X[0])
        n_classes = len(set(self.y_labels))

        if name == "Perceptron":
            self.perceptron = Perceptron(input_dim, n_classes, lr)
            self.perceptron.train(self.X, self.y_idx, epochs=100)
            self.current_model_name = "Perceptron"
        else:
            self.adaline = Adaline(input_dim, n_classes, lr)
            self.adaline.train(self.X, self.y_idx, epochs=200)
            self.current_model_name = "Adaline"

        self.status_label.config(text=f"{name} training completed!", fg="black")

    def classify_drawing(self):
        if self.current_model_name is None:
            messagebox.showwarning("No model", "Please train a model before classifying.")
            return
        x = bipolarize(self.grid_state).flatten()
        pred_idx = (self.perceptron if self.current_model_name == "Perceptron" else self.adaline).predict(x)
        self.result_label.config(text=f"Recognized: {self.y_labels[pred_idx]}")

    def save_example(self):
        label = self.save_label_var.get()
        x = bipolarize(self.grid_state).flatten()
        self.X.append(x)
        self.y_labels.append(label)
        self.y_idx.append(self.y_labels.index(label))
        messagebox.showinfo("Saved", f"Example saved as '{label}'.")


if __name__ == "__main__":
    def launch_gui():
        root = tk.Tk()
        app = ANNapp(root)
        root.mainloop()

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
• Adaline
• Hebb
• Perceptron

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