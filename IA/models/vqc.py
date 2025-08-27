import pennylane as qml

class VQC():
    def __init__(self, n_qubits:int = None, n_layers:int = 3, data_shape:tuple = ()):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        if self.n_qubits > data_shape[1]:
            print("Numero de qubits deve ser <= que o numero de dimensoes do dataset")
            return
        elif self.n_qubits == data_shape[1]:
            print("Numero de qubits eh exatamente igual ao numero de dimensoes do dataset")
        else:
            print("Numero de qubits eh menor que o numero de dimensoes do dataset")

        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
    # quantum circuit functions
    def statepreparation(self, x):
        #qml.BasisEmbedding(x, wires=range(0, num_qubits))
        qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation='Y')

    def layer(self, W):

        qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
        qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
        qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
        qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

        if self.n_qubits >= 2:
            qml.CNOT(wires=[0, 1])
        if self.n_qubits >= 3:
            qml.CNOT(wires=[1, 2])
        if self.n_qubits >= 4:
            qml.CNOT(wires=[2, 3])
        if self.n_qubits >= 1 and self.n_qubits != 0: # Ensure at least 1 qubit before this
            qml.CNOT(wires=[self.n_qubits - 1, 0]) # Wrap around CNOT

    @qml.qnode(dev, interface="autograd")
    def circuit(self, weights, x):

        self.statepreparation(x)

        for W in weights:
            self.layer(W)

        return qml.expval(qml.PauliZ(0))

    def variational_classifier(self, weights, bias, x):
        return self.circuit(weights, x) + bias

    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2

        loss = loss / len(labels)
        return loss

    def accuracy(self, labels, predictions):

        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)

        return loss

    def cost(self, weights, bias, X, Y):
        predictions = [self.variational_classifier(weights, bias, x) for x in X]
        return self.square_loss(Y, predictions)