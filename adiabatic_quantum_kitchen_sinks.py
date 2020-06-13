import numpy as np
import neal
from quantum_kitchen_sink import QuantumKitchenSink

class AdiabaticQuantumKitchenSink(QuantumKitchenSink):
    def run_quantum_program(self, theta):
        j_e = theta
        h = {}
        for i, value  in enumerate(theta):
            h[i] = value

        j = {}
        for i in range(len(theta)):
            for k in range(len(theta)):
                j[(i, k)] = theta[i] * theta[k]

        SA_sampler = neal.SimulatedAnnealingSampler()
        samples = SA_sampler.sample_ising(h=h, J=j, num_reads=self.shots)
        means = np.zeros(self.q)
        for sample in samples:
            for k, v in sample.items():
                if sample[k] == 1:
                    means[k] += 1

        means = np.array(means)/self.shots
        return means