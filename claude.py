"""
ANN Numbers' Classifier
=======================
Algorithms : Hebb Rule | Perceptron | Adaline (Delta Rule)
Grid       : 10 rows × 7 cols = 70 inputs
Numbers    : "0" to "7"  (patterns from Settings.py)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from Settings import numbers_set          # ← your original Settings.py

def bipolarize(grid):
    """Convert binary 0/1 list/array to bipolar -1/+1 numpy array."""
    return np.array(grid, dtype=float) * 2 - 1

LABELS   = list(numbers_set.keys())                              # ["0".."7"]
X_PROTO  = np.array([bipolarize(numbers_set[k]).flatten()
                     for k in LABELS])                           # (8, 70)
N_IN     = X_PROTO.shape[1]   # 70  (10 rows × 7 cols)
N_CLS    = len(LABELS)         # 8

T_PROTO  = np.where(np.arange(N_CLS)[:, None] == np.arange(N_CLS),
                    1.0, -1.0)                                   # (8, 8)

class Hebb:
    def __init__(self):
        self.W = None                       # weight matrix (n_classes × n_inputs)

    def train(self, X, T):
        """
        W = Tᵀ · X   (matrix form of  Σ gₖ · fₖᵀ  over all patterns)
        X : (n_samples, n_inputs)   — bipolar input patterns
        T : (n_samples, n_classes)  — bipolar one-hot targets
        """
        self.W = T.T @ X                    # shape: (n_classes, n_inputs)

    def predict(self, x):
        """argmax(W · x) — the most activated output neuron wins."""
        return int(np.argmax(self.W @ x))

class Perceptron:
    def __init__(self, lr=0.2):
        self.lr = lr
        self.W  = None
        self.b  = None
        self.errors_per_epoch = []          # useful for convergence plot

    def train(self, X, T, epochs=200):
        """
        X : (n_samples, n_inputs)
        T : (n_samples, n_classes)  — bipolar one-hot targets
        """
        self.W = np.zeros((N_CLS, N_IN))
        self.b = np.zeros(N_CLS)
        self.errors_per_epoch = []

        for _ in range(epochs):
            n_wrong = 0
            for xi, ti in zip(X, T):
                net   = self.W @ xi + self.b        # net activation
                yi    = np.sign(net)                # step activation
                yi[yi == 0] = 1                     # sign(0) = +1 by convention
                delta = ti - yi                     # error: 0 if correct, ±2 if wrong
                if np.any(delta != 0):              # update ONLY on error
                    self.W += self.lr * np.outer(delta, xi)
                    self.b += self.lr * delta
                    n_wrong += 1
            self.errors_per_epoch.append(n_wrong)
            if n_wrong == 0:                        # converged — stop early
                break

    def predict(self, x):
        return int(np.argmax(self.W @ x + self.b))

class Adaline:
    def __init__(self, lr=0.001):
        self.lr = lr
        self.W  = None
        self.b  = None
        self.mse_per_epoch = []             # useful for convergence plot

    def train(self, X, T, epochs=500):
        """
        X : (n_samples, n_inputs)
        T : (n_samples, n_classes)  — bipolar one-hot targets
        """
        self.W = np.zeros((N_CLS, N_IN))
        self.b = np.zeros(N_CLS)
        self.mse_per_epoch = []

        for _ in range(epochs):
            epoch_mse = 0.0
            for xi, ti in zip(X, T):
                net        = self.W @ xi + self.b   # LINEAR — no sign!
                delta      = ti - net               # error on linear output
                self.W    += self.lr * np.outer(delta, xi)
                self.b    += self.lr * delta
                epoch_mse += np.mean(delta ** 2)
            self.mse_per_epoch.append(epoch_mse / len(X))

    def predict(self, x):
        return int(np.argmax(self.W @ x + self.b))

class ANNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Numbers' Classifier  —  Hebb | Perceptron | Adaline")

        self.rows      = 10
        self.cols      = 7
        self.cell_size = 50

        self.models = {"Hebb": Hebb(), "Perceptron": Perceptron(), "Adaline": Adaline()}

        self.X_extra = []
        self.T_extra = []

        self.grid_state = np.zeros((self.rows, self.cols), dtype=int)

        self._build_ui()

    def _build_ui(self):
        ctrl = tk.Frame(self.root, padx=10, pady=10)
        ctrl.grid(row=0, column=0, sticky="n")

        tk.Label(ctrl, text="Algorithm", font=("Arial", 10, "bold")).pack(anchor="w")
        self.algo_var = tk.StringVar(value="Perceptron")
        for name in ["Hebb", "Perceptron", "Adaline"]:
            tk.Radiobutton(ctrl, text=name, variable=self.algo_var, value=name).pack(anchor="w")

        tk.Label(ctrl, text="\nLearning Rate").pack(anchor="w")
        self.lr_var = tk.StringVar(value="0.2")
        tk.Entry(ctrl, textvariable=self.lr_var, width=10).pack(anchor="w")

        tk.Label(ctrl, text="Epochs").pack(anchor="w")
        self.ep_var = tk.IntVar(value=200)
        tk.Entry(ctrl, textvariable=self.ep_var, width=10).pack(anchor="w")

        tk.Button(ctrl, text="Train Algorithm",
                  command=self._train, bg="#4472C4", fg="white", width=20).pack(pady=(8, 2))
        tk.Button(ctrl, text="Classify Drawing",
                  command=self._classify, bg="#70AD47", fg="white", width=20).pack(pady=2)
        tk.Button(ctrl, text="Clear Canvas",
                  command=self._clear, width=20).pack(pady=2)

        tk.Label(ctrl, text="\nLoad prototype digit:").pack(anchor="w")
        self.load_var = tk.StringVar(value="0")
        ttk.Combobox(ctrl, textvariable=self.load_var,
                     values=LABELS, state="readonly", width=8).pack(anchor="w")
        tk.Button(ctrl, text="Load →",
                  command=self._load_digit, width=20).pack(pady=2)

        tk.Label(ctrl, text="\nSave drawing as:").pack(anchor="w")
        self.save_var = tk.StringVar(value="0")
        ttk.Combobox(ctrl, textvariable=self.save_var,
                     values=LABELS, state="readonly", width=8).pack(anchor="w")
        tk.Button(ctrl, text="Save as Example",
                  command=self._save_example, width=20).pack(pady=2)

        tk.Label(ctrl, text="\nResult", font=("Arial", 10, "bold")).pack()
        self.result_lbl = tk.Label(ctrl, text="—",
                                   font=("Arial", 40, "bold"), fg="#C00000")
        self.result_lbl.pack()
        self.status_lbl = tk.Label(ctrl, text="",
                                   font=("Arial", 9), fg="gray", wraplength=185)
        self.status_lbl.pack()

        cf = tk.Frame(self.root, bg="#333", padx=2, pady=2)
        cf.grid(row=0, column=1, padx=10, pady=10)

        self.canvas = tk.Canvas(cf,
                                width=self.cols * self.cell_size,
                                height=self.rows * self.cell_size,
                                bg="white", cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>",  self._toggle_cell)   # click = toggle
        self.canvas.bind("<B1-Motion>", self._paint_cell)    # drag  = paint

        self.rects = []
        for r in range(self.rows):
            row_rects = []
            for c in range(self.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                rect = self.canvas.create_rectangle(
                    x1, y1, x1 + self.cell_size, y1 + self.cell_size,
                    fill="white", outline="#aaa")
                row_rects.append(rect)
            self.rects.append(row_rects)

    def _cell_at(self, event):
        """Return (row, col) for a mouse event, or None if out of bounds."""
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def _set_cell(self, r, c, value):
        """Set cell (r,c) to value (0 or 1) and update canvas colour."""
        self.grid_state[r, c] = value
        self.canvas.itemconfig(self.rects[r][c],
                               fill="black" if value else "white")

    def _toggle_cell(self, event):
        pos = self._cell_at(event)
        if pos:
            r, c = pos
            self._set_cell(r, c, 1 - self.grid_state[r, c])

    def _paint_cell(self, event):
        pos = self._cell_at(event)
        if pos:
            self._set_cell(*pos, 1)

    def _clear(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self._set_cell(r, c, 0)
        self.result_lbl.config(text="—")
        self.status_lbl.config(text="Canvas cleared.")

    def _load_digit(self):
        """Paint the stored prototype for the selected digit."""
        label = self.load_var.get()
        self._clear()
        grid = numbers_set[label]           # original binary 0/1 from Settings
        for r in range(self.rows):
            for c in range(self.cols):
                self._set_cell(r, c, grid[r][c])
        self.status_lbl.config(text=f"Loaded prototype for digit '{label}'.")

    def _get_training_data(self):
        """Return the full training set (prototypes + any saved examples)."""
        if self.X_extra:
            X = np.vstack([X_PROTO] + self.X_extra)
            T = np.vstack([T_PROTO] + self.T_extra)
        else:
            X, T = X_PROTO, T_PROTO
        return X, T

    def _train(self):
        algo = self.algo_var.get()
        try:
            lr = float(self.lr_var.get())
            ep = int(self.ep_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Learning rate and epochs must be numbers.")
            return

        X, T = self._get_training_data()
        self.status_lbl.config(text=f"Training {algo}…", fg="gray")
        self.root.update()

        if algo == "Hebb":
            self.models["Hebb"] = Hebb()
            self.models["Hebb"].train(X, T)
            self.status_lbl.config(
                text="Hebb trained in 1 pass  (no iterations needed).", fg="black")

        elif algo == "Perceptron":
            self.models["Perceptron"] = Perceptron(lr=lr)
            self.models["Perceptron"].train(X, T, epochs=ep)
            n_ep  = len(self.models["Perceptron"].errors_per_epoch)
            final = self.models["Perceptron"].errors_per_epoch[-1]
            self.status_lbl.config(
                text=f"Perceptron: converged in {n_ep} epoch(s), final errors = {final}.",
                fg="black")

        elif algo == "Adaline":
            safe_lr = min(lr, 0.005)            # prevent divergence
            self.models["Adaline"] = Adaline(lr=safe_lr)
            self.models["Adaline"].train(X, T, epochs=ep)
            final_mse = self.models["Adaline"].mse_per_epoch[-1]
            self.status_lbl.config(
                text=f"Adaline: {ep} epochs, MSE = {final_mse:.6f}  (lr = {safe_lr}).",
                fg="black")

    def _classify(self):
        algo  = self.algo_var.get()
        model = self.models[algo]

        if (algo == "Hebb"       and model.W is None) or \
           (algo != "Hebb"       and model.W is None):
            messagebox.showwarning("Not trained",
                                   f"Please train {algo} before classifying.")
            return

        x    = bipolarize(self.grid_state).flatten()   # (70,) bipolar
        pred = model.predict(x)
        self.result_lbl.config(text=LABELS[pred])
        self.status_lbl.config(
            text=f"[{algo}] → digit '{LABELS[pred]}'.", fg="black")

    def _save_example(self):
        """Add the current drawing to the training set."""
        if not np.any(self.grid_state):
            messagebox.showwarning("Empty canvas", "Draw something first.")
            return
        label = self.save_var.get()
        idx   = LABELS.index(label)

        x = bipolarize(self.grid_state).flatten()
        t = np.full(N_CLS, -1.0)
        t[idx] = 1.0

        self.X_extra.append(x)
        self.T_extra.append(t)

        messagebox.showinfo(
            "Saved",
            f"Drawing saved as digit '{label}'.\n"
            f"Training set now has {len(X_PROTO) + len(self.X_extra)} examples.\n"
            f"Re-train to apply the new example.")

if __name__ == "__main__":
    print("""
══════════════════════════════════════════════════════════════
          ANN NUMBERS' CLASSIFIER
    Hebb  ·  Perceptron  ·  Adaline (Delta Rule)
══════════════════════════════════════════════════════════════
Supported digits : 0 – 7
Grid size        : 10 rows × 7 cols = 70 inputs

HOW TO USE:
  1. Select an algorithm (radio buttons).
  2. Click "Train Algorithm".
  3. Draw a digit on the canvas (click or drag).
  4. Click "Classify Drawing" to get the prediction.
  5. Use "Load →" to test with a stored prototype.
  6. Use "Save as Example" to add your drawing to the
     training set, then retrain.

ALGORITHM NOTES:
  Hebb      — no learning rate, trains in one pass.
  Perceptron— lr=0.2, epochs=200 recommended.
  Adaline   — lr is auto-capped at 0.005 for stability.
══════════════════════════════════════════════════════════════
""")
    root = tk.Tk()
    ANNApp(root)
    root.mainloop()