import numpy as np

class QuantumKitchenSink():
    def __init__(self, episodes, q, r = 1, shots = 100):
        self.episodes = episodes
        self.r = r
        self.q = q
        self.shots = shots
        self.cols = episodes * q

    def fit_transform(self, X):
        self.p = X.shape[-1]
        size = (self.episodes, self.q, self.p)
        distribution = np.random.normal(loc=0, scale=1, size=size)
        selection_matrix = self.generate_selection_matrix()

        omega = distribution * selection_matrix
        beta = np.random.uniform(low=0, high=(2 * np.pi), size=(self.episodes, self.q))

        theta = []

        for i in range(X.shape[0]):
            for e in range(self.episodes):
                theta.append(omega[e].dot(X[i].T) + beta[e])

        theta = np.array(theta)

        transforms = []
        for t in theta:
            measurements = self.run_quantum_program(t)
            transforms.append(measurements)
        return np.array(transforms).reshape(X.shape[0], self.cols)

    def generate_selection_matrix(self):
        return np.array([self.build_matrix() for i in range(self.episodes)])

    def build_matrix(self):
        m_size = self.p * self.q
        m = np.zeros(m_size)
        for i in range(self.r):
            m[i] = 1
        np.random.shuffle(m)
        return m.reshape(self.q, self.p)

    def run_quantum_program(self, theta):
        pass