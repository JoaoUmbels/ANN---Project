import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np




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


class WidrowHoff:
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


class ANNClassifierApp:
    def __init__(self, root):
        self.root = root
        root.title("ANN Character Classifier")

        self.rows, self.cols = 9, 7
        self.cell_size = 40

        self.char_patterns = default_char_patterns.copy()
        self.patterns = {k: bipolarize(v).flatten() for k, v in self.char_patterns.items()}
        self.X = list(self.patterns.values())
        self.y_labels = list(self.patterns.keys())
        self.y_idx = list(range(len(self.y_labels)))

        self.perceptron = None
        self.widrow = None
        self.current_model_name = None


        self.canvas = tk.Canvas(root, width=self.cols * self.cell_size, height=self.rows * self.cell_size, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=6, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        self.grid_state = np.zeros((self.rows, self.cols), dtype=int)
        self.rects = []
        for r in range(self.rows):
            row_rects = []
            for c in range(self.cols):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="black")
                row_rects.append(rect)
            self.rects.append(row_rects)

        # UI Controls
        self.algorithm_var = tk.StringVar(value="Perceptron")
        ttk.Combobox(root, textvariable=self.algorithm_var, values=["Perceptron", "Widrow-Hoff"],
                     state="readonly").grid(row=1, column=0, padx=5)
        self.lr_var = tk.StringVar(value="0.2")
        tk.Entry(root, textvariable=self.lr_var, width=7).grid(row=1, column=1, padx=5)
        self.save_label_var = tk.StringVar(value=self.y_labels[0])
        ttk.Combobox(root, textvariable=self.save_label_var, values=self.y_labels, state="readonly").grid(row=1,
                                                                                                          column=2,
                                                                                                          padx=5)

        tk.Button(root, text="Train Model", command=self.train_model).grid(row=2, column=0, columnspan=1, sticky="ew",
                                                                           padx=5, pady=5)
        tk.Button(root, text="Classify Drawing", command=self.classify_drawing).grid(row=2, column=1, columnspan=1,
                                                                                     sticky="ew", padx=5, pady=5)
        tk.Button(root, text="Save as Example", command=self.save_example).grid(row=2, column=2, columnspan=1,
                                                                                sticky="ew", padx=5, pady=5)
        tk.Button(root, text="Clear Drawing", command=self.clear_drawing).grid(row=3, column=0, columnspan=3,
                                                                               sticky="ew", padx=5, pady=5)

        self.status_label = tk.Label(root, text="Train a model to start.", fg="blue")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.grid(row=5, column=0, columnspan=3, pady=5)

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

        algo = self.algorithm_var.get()
        self.status_label.config(text=f"Training {algo}...", fg="gray")
        self.root.update()

        input_dim = len(self.X[0])
        n_classes = len(set(self.y_labels))

        if algo == "Perceptron":
            self.perceptron = Perceptron(input_dim, n_classes, lr)
            self.perceptron.train(self.X, self.y_idx, epochs=100)
            self.current_model_name = "Perceptron"
        else:
            self.widrow = WidrowHoff(input_dim, n_classes, lr)
            self.widrow.train(self.X, self.y_idx, epochs=200)
            self.current_model_name = "Widrow-Hoff"

        self.status_label.config(text=f"{algo} training completed!", fg="black")

    def classify_drawing(self):
        if self.current_model_name is None:
            messagebox.showwarning("No model", "Please train a model before classifying.")
            return
        x = bipolarize(self.grid_state).flatten()
        pred_idx = (self.perceptron if self.current_model_name == "Perceptron" else self.widrow).predict(x)
        pred_label = self.y_labels[pred_idx]
        self.result_label.config(text=f"Recognized character: {pred_label}")

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
        app = ANNClassifierApp(root)
        root.mainloop()


    if __name__ == "__main__":
        print("=============================")
        print("WELCOME TO THE ANN PROJECT!!!")
        print("==============================\n")
        print("This application classifies characters (A, B, C, D, E, J, K)")
        print("Using Perceptron and Widrow-Hoff learning algorithms.\n")
        print("HOW TO USE:")
        print("- You will be shown a canvas window.")
        print('- Choose and TRAIN the algorithm of your choice (Perceptron or Widrow-Hoff)')
        print("- Use your mouse to draw one of the letters: A, B, C, D, E, J, or K.")
        print("- Then click in the 'Classify Drawing' button to see the prediction.")
        print("- To SAVE the current character, select one of the letters and click the 'Save as example' Button ")
        print("- You can clear and draw again as many times as you'd like.\n")
        input("Press ENTER to open the drawing window...")
        launch_gui()
