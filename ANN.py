import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from Settings import numbers_set
import time

def bipolarize(arr): # Converts binary (0/1) to bipolar (-1/+1): value * 2 - 1
    return np.array(arr) * 2 - 1

class Hebb: 
    def __init__(self):
        self.W = None  # Weight matrix  (n_classes × n_inputs)
        self.b = None  # Bias vector    (n_classes,)

    def train(self, X, y_idx):
        start_t = time.time()
        total_correct = 0
        total_error = 0

        n_numbers = len(set(y_idx))
        n_inputs = len(X[0])
        self.W = np.zeros((n_numbers, n_inputs))
        self.b = np.zeros(n_numbers)

        for xi, target in zip(X, y_idx):
            for i in range(n_numbers):
                desired = 1 if i == target else -1
                self.W[i] += desired * xi
                self.b[i] += desired

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

    def predict(self, x): # Returns class with highest net activation: argmax(W·x + b)
        outputs = np.dot(self.W, x) + self.b
        return np.argmax(outputs)
    
    


class Perceptron: # w(new) = w(old) + α·error·xi,  stops when no errors in a full pass
    def __init__(self, lr=0.2):
        self.W = None
        self.b = None
        self.lr = lr

    def train(self, X, y_idx, iterations=100):
        start_t = time.time()
        total_correct = 0
        total_error = 0
        n_numbers = len(set(y_idx))
        n_inputs = len(X[0])
        self.W = np.zeros((n_numbers, n_inputs))
        self.b = np.zeros(n_numbers)
        epochs_run = 0
        for _ in range(iterations):
            epochs_run += 1
            has_error = False
            for xi, target in zip(X, y_idx):
                for i in range(n_numbers):
                    out = np.sign(np.dot(self.W[i], xi) + self.b[i])
                    desired = 1 if i == target else -1
                    error = desired - out
                    if error != 0:
                        self.W[i] += self.lr * error * xi
                        self.b[i] += self.lr * error
                        has_error = True

            if not has_error:  # Converged — no errors in this epoch
                break

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
    
    def predict(self, x): # Returns class with highest net activation: argmax(W·x + b)
        outputs = np.dot(self.W, x) + self.b
        return np.argmax(outputs)

class Adaline:
    def __init__(self, lr=0.001):
        self.W = None
        self.b = None
        self.lr = lr

    def train(self, X, y_idx, epochs=200):
        start_t = time.time()
        total_correct = 0
        total_error = 0
        n_numbers = len(set(y_idx))
        n_inputs = len(X[0])
        self.W = np.zeros((n_numbers, n_inputs))
        self.b = np.zeros(n_numbers)
        final_mse = 0
        for _ in range(epochs):
            W_gradient = np.zeros_like(self.W)
            b_gradient = np.zeros_like(self.b)
            epoch_mse = 0

            for xi, target in zip(X, y_idx):
                desired = np.full(n_numbers, -1.0)
                desired[target] = 1.0
                net = np.dot(self.W, xi) + self.b   # Linear activation (y_in)
                error = desired - net                 # Delta: t - y_in
                epoch_mse += np.sum(error ** 2)
                W_gradient += np.outer(error, xi)    # Accumulate ΔW
                b_gradient += error                  # Accumulate Δb
 
            final_mse = epoch_mse / (len(X) * n_numbers)
            self.W += self.lr * W_gradient           # Batch weight update
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

    def predict(self, x): # Returns class with highest net activation: argmax(W·x + b)
        outputs = np.dot(self.W, x) + self.b
        return np.argmax(outputs)

class ANNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Numbers Classifier")

        self.rows = 10
        self.cols = 7
        self.cell_size = 50

        self.draws = {k: bipolarize(v).flatten() for k, v in numbers_set.items()}
        self.X = list(self.draws.values())
        self.y_labels = list(self.draws.keys())
        self.y_idx = list(range(len(self.y_labels)))

        self.hebb = None
        self.perceptron = None
        self.adaline = None
        self.current_model = None

        self.grid_state = np.zeros((self.rows, self.cols), dtype=int)

        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size,
                                height=self.rows * self.cell_size, bg="white")
        self.canvas.grid(row=0, column=1, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.rects = []
        for r in range(self.rows):
            row_rects = []
            for c in range(self.cols):
                x1, y1 = c * self.cell_size, r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                    fill="white", outline="black")
                row_rects.append(rect)
            self.rects.append(row_rects)

        self.create_controls()

    def create_controls(self):
        controls = tk.Frame(self.root, padx=10, pady=10)
        controls.grid(row=0, column=0, sticky="n")

        tk.Label(controls, text="Algorithm").pack(anchor="w")
        self.model_var = tk.StringVar(value="Perceptron")
        self.model_choice_box = ttk.Combobox(controls, textvariable=self.model_var,
                     values=["Hebb", "Perceptron", "Adaline"],
                     state="readonly", width=18)
        self.model_choice_box.pack()
        self.model_choice_box.bind("<<ComboboxSelected>>", self.update_ui_visibility)

        self.hyperparams_frame = tk.Frame(controls)
        self.hyperparams_frame.pack(fill="x")

        self.epochs_frame = tk.Frame(self.hyperparams_frame)
        tk.Label(self.epochs_frame, text="Epochs").pack(anchor="w")
        self.epochs = tk.StringVar(value="200")
        tk.Entry(self.epochs_frame, textvariable=self.epochs, width=10).pack()

        self.lr_frame = tk.Frame(self.hyperparams_frame)
        tk.Label(self.lr_frame, text="Learning Rate").pack(anchor="w")
        self.lr_var = tk.StringVar(value="0.2")
        tk.Entry(self.lr_frame, textvariable=self.lr_var, width=10).pack()

        tk.Label(controls, text="Number Label").pack(anchor="w")
        self.save_label_var = tk.StringVar(value=self.y_labels[0])
        ttk.Combobox(controls, textvariable=self.save_label_var,
                     values=self.y_labels, state="readonly", width=18).pack()

        tk.Button(controls, text="Train Model", command=self.train_model).pack(fill="x", pady=2)
        tk.Button(controls, text="Classify Drawing", command=self.classify_drawing).pack(fill="x", pady=2)
        tk.Button(controls, text="Save as Example", command=self.save_example).pack(fill="x", pady=2)
        tk.Button(controls, text="Clear Drawing", command=self.clear_drawing).pack(fill="x", pady=2)

        tk.Button(root, text="📊 View Performance Dashboard", command=self.show_dashboard, bg="#d1ecf1").grid(row=3, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        self.result_label = tk.Label(controls, text="")
        self.result_label.pack()

        self.status_label = tk.Label(controls, text="", fg="gray")
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

    def on_click(self, event):
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
        name = self.model_var.get()

        lr = 0.2
        if name in ["Perceptron", "Adaline"]:
            try:
                lr = float(self.lr_var.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Learning rate must be a number.")
                return
        self.status_label.config(text=f"Training {name}...", fg="gray")
        self.root.update()

        if name == "Hebb":
            self.hebb = Hebb()
            statistics = self.hebb.train(self.X, self.y_idx)
            self.current_model = "Hebb"
            self.performance_history["Hebb"] = statistics

        elif name == "Perceptron":
            self.perceptron = Perceptron(lr)
            statistics = self.perceptron.train(self.X, self.y_idx)
            self.current_model = "Perceptron"
            self.performance_history["Perceptron"] = statistics

        else:
            self.adaline = Adaline(lr)
            try:
                ep = int(self.epochs.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Epochs must be an integer.")
                self.status_label.config(text="Training failed.", fg="red")
                return
            statistics = self.adaline.train(self.X, self.y_idx, ep)
            self.current_model = "Adaline"
            self.performance_history["Adaline"] = statistics

        self.status_label.config(text=f"{name} training completed!", fg="black")

    def classify_drawing(self):
        if self.current_model is None:
            messagebox.showwarning("No model", "Please train a model first.")
            return

        x = bipolarize(self.grid_state).flatten()

        if self.current_model == "Hebb":
            pred = self.hebb.predict(x)
        elif self.current_model == "Perceptron":
            pred = self.perceptron.predict(x)
        else:
            pred = self.adaline.predict(x)

        self.result_label.config(text=f"Recognized: {self.y_labels[pred]}")

    def save_example(self):
        label = self.save_label_var.get()
        x = bipolarize(self.grid_state).flatten()
        self.X.append(x)
        self.y_labels.append(label)
        self.y_idx.append(self.y_labels.index(label))
        messagebox.showinfo("Saved", f"Example saved as '{label}'.")

    def show_dashboard(self):
        dash_window = tk.Toplevel(self.root)
        dash_window.title("Algorithm Performance Evaluation")
        dash_window.geometry("800x300")
        
        tk.Label(dash_window, text="Algorithm Comparison Matrix", font=("Arial", 14, "bold")).pack(pady=10)
        
        columns = ("metric", "hebb", "perceptron", "adaline")
        tree = ttk.Treeview(dash_window, columns=columns, show="headings", height=7)
        tree.pack(fill="both", expand=True, padx=10, pady=5)
        
        tree.heading("metric", text="Evaluation Topic")
        tree.heading("hebb", text="Hebb")
        tree.heading("perceptron", text="Perceptron")
        tree.heading("adaline", text="Adaline (Widrow-Hoff)")
        
        tree.column("metric", width=180, anchor="center")
        tree.column("hebb", width=150, anchor="center")
        tree.column("perceptron", width=150, anchor="center")
        tree.column("adaline", width=180, anchor="center")
        
        h_stats = self.performance_history["Hebb"]
        p_stats = self.performance_history["Perceptron"]
        w_stats = self.performance_history["Adaline"]
        
        def fmt(val, suffix="", precision=4):
            return f"{val:.{precision}f}{suffix}" if isinstance(val, (int, float)) else val

        tree.insert("", "end", values=("Training Time", fmt(h_stats['time'], " sec"), fmt(p_stats['time'], " sec"), fmt(w_stats['time'], " sec")))
        tree.insert("", "end", values=("Epochs", fmt(h_stats['epochs'], precision=0), fmt(p_stats['epochs'], precision=0), fmt(w_stats['epochs'], precision=0)))
        tree.insert("", "end", values=("Iterations", fmt(h_stats['iterations'], precision=0), fmt(p_stats['iterations'], precision=0), fmt(w_stats['iterations'], precision=0)))
        tree.insert("", "end", values=("Final MSE", fmt(h_stats['mse']), fmt(p_stats['mse']), fmt(w_stats['mse'])))
        tree.insert("", "end", values=("Correct", fmt(h_stats['correct'], precision=0), fmt(p_stats['correct'], precision=0), fmt(w_stats['correct'])))
        tree.insert("", "end", values=("Errors", fmt(h_stats['wrong'], precision=0), fmt(p_stats['wrong'], precision=0), fmt(w_stats['wrong'])))
        tree.insert("", "end", values=("Train Dataset Accuracy", fmt(h_stats['accuracy'], 1), fmt(p_stats['accuracy'], 1), fmt(w_stats['accuracy'], 1)))
        
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
    
    root = tk.Tk()
    app = ANNApp(root)
    root.mainloop()