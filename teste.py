import tkinter as tk
import numpy as np

# === Multi-class Perceptron ===
class Perceptron:
    def __init__(self, input_size, lr=0.1, num_classes=7):
        self.lr = lr
        self.weights = np.zeros((num_classes, input_size + 1))  # One row per class

    def predict(self, x):
        activations = np.dot(self.weights[:, 1:], x) + self.weights[:, 0]
        return np.argmax(activations)  # Return index with highest activation

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                if pred != target:
                    # Increase weights for correct class
                    self.weights[target, 1:] += self.lr * xi
                    self.weights[target, 0] += self.lr
                    # Decrease weights for wrong prediction
                    self.weights[pred, 1:] -= self.lr * xi
                    self.weights[pred, 0] -= self.lr

# === Multi-class Widrow-Hoff (Delta Rule) ===
class WidrowHoff:
    def __init__(self, input_size, lr=0.01, num_classes=7):
        self.lr = lr
        self.weights = np.random.rand(num_classes, input_size + 1)

    def predict(self, x):
        outputs = np.dot(self.weights[:, 1:], x) + self.weights[:, 0]
        return np.argmax(outputs)

    def train(self, X, y, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                outputs = np.dot(self.weights[:, 1:], xi) + self.weights[:, 0]
                targets = np.zeros_like(outputs)
                targets[target] = 1
                error = targets - outputs
                self.weights[:, 1:] += self.lr * error[:, np.newaxis] * xi
                self.weights[:, 0] += self.lr * error

# === GUI Application ===
class ANNApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN Character Classifier")

        # Labels: A, B, C, D, E, J, K
        self.labels = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        self.label_to_index = {label: i for i, label in enumerate(self.labels)}

        self.train_X = []
        self.train_y = []

        # GUI variables
        self.algorithm = tk.StringVar(value="Perceptron")
        self.lr = tk.DoubleVar(value=0.1)
        self.epochs = tk.IntVar(value=10)
        self.char_label = tk.StringVar(value="A")
        self.model = None

        self.create_controls()
        self.create_grid()

    def create_controls(self):
        tk.Label(self.root, text="Algorithm").pack()
        tk.OptionMenu(self.root, self.algorithm, "Perceptron", "Widrow-Hoff").pack()

        tk.Label(self.root, text="Learning Rate").pack()
        tk.Entry(self.root, textvariable=self.lr).pack()

        tk.Label(self.root, text="Epochs").pack()
        tk.Entry(self.root, textvariable=self.epochs).pack()

        tk.Label(self.root, text="Character Label (A–K)").pack()
        tk.Entry(self.root, textvariable=self.char_label).pack()

        tk.Button(self.root, text="Add Sample", command=self.add_sample).pack(pady=5)
        tk.Button(self.root, text="Train", command=self.train_model).pack(pady=5)
        tk.Button(self.root, text="Classify", command=self.classify_input).pack(pady=5)

        self.result = tk.Label(self.root, text="", fg="blue")
        self.result.pack(pady=10)

    def create_grid(self):
        self.grid = []
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack()

        for i in range(5):
            row = []
            for j in range(5):
                btn = tk.Button(self.grid_frame, width=2, height=1, bg="white",
                                command=lambda i=i, j=j: self.toggle_pixel(i, j))
                btn.grid(row=i, column=j, padx=2, pady=2)
                row.append(btn)
            self.grid.append(row)

    def toggle_pixel(self, i, j):
        btn = self.grid[i][j]
        current_color = btn.cget("bg")
        btn.config(bg="black" if current_color == "white" else "white")

    def get_current_grid_input(self):
        return np.array([
            1 if self.grid[i][j].cget("bg") == "black" else 0
            for i in range(5) for j in range(5)
        ])

    def add_sample(self):
        x = self.get_current_grid_input()
        label = self.char_label.get().upper()
        if label not in self.label_to_index:
            self.result.config(text="Use only A, B, C, D, E, J, or K.")
            return
        y = self.label_to_index[label]
        self.train_X.append(x)
        self.train_y.append(y)
        self.clear_grid()
        self.result.config(text=f"Sample for '{label}' added.")

    def train_model(self):
        if not self.train_X:
            self.result.config(text="No training data yet.")
            return
        X = np.array(self.train_X)
        y = np.array(self.train_y)
        input_size = X.shape[1]
        num_classes = len(self.labels)

        if self.algorithm.get() == "Perceptron":
            self.model = Perceptron(input_size, self.lr.get(), num_classes)
        else:
            self.model = WidrowHoff(input_size, self.lr.get(), num_classes)

        self.model.train(X, y, self.epochs.get())
        self.result.config(text="Training complete.")

    def classify_input(self):
        if not self.model:
            self.result.config(text="Model not trained yet.")
            return
        x = self.get_current_grid_input()
        prediction = self.model.predict(x)
        predicted_label = self.labels[prediction]
        self.result.config(text=f"Prediction: {predicted_label}")

    def clear_grid(self):
        for row in self.grid:
            for btn in row:
                btn.config(bg="white")

# === Main ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ANNApp(root)
    root.mainloop()

                
