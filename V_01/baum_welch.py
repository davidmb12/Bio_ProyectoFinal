import math
from collections import defaultdict
import numpy as np

# Supongamos que ya tienes implementadas funciones para:
# - forward(hmm, sequence): que devuelve una matriz alpha para la secuencia.
# - backward(hmm, sequence): que devuelve una matriz beta para la secuencia.
# Aquí se asume que "hmm" es un objeto o diccionario que contiene:
#     - hmm.states: lista de estados
#     - hmm.trans_probs: probabilidades de transición, por ejemplo, hmm.trans_probs[i][j]
#     - hmm.emiss_probs: probabilidades de emisión, por ejemplo, hmm.emiss_probs[state][símbolo]

def forward(hmm, observations):
    T = len(observations)
    N = len(hmm.states)
    alpha = np.zeros((T, N))
    # Inicialización: asumiendo que se inicia en un estado especial "S" que transita a otros.
    # Aquí hay que adaptar según cómo se haya modelado el estado de inicio.
    # Por ejemplo, podrías tener una distribución inicial en hmm.init_probs.
    for i, state in enumerate(hmm.states):
        # Usar distribución inicial (suponiendo que está en hmm.init_probs)
        alpha[0, i] = hmm.init_probs.get(state, 0) * hmm.emiss_probs.get(state, {}).get(observations[0], 0)
    
    # Recursión
    for t in range(1, T):
        for j, state_j in enumerate(hmm.states):
            s = 0
            for i, state_i in enumerate(hmm.states):
                s += alpha[t-1, i] * hmm.trans_probs.get(state_i, {}).get(state_j, 0)
            alpha[t, j] = s * hmm.emiss_probs.get(state_j, {}).get(observations[t], 0)
    return alpha

def backward(hmm, observations):
    T = len(observations)
    N = len(hmm.states)
    beta = np.zeros((T, N))
    # Inicialización: en el tiempo final, probabilidad 1 para estados que transitan al estado final E
    beta[T-1, :] = 1
    # Recursión hacia atrás
    for t in reversed(range(T-1)):
        for i, state_i in enumerate(hmm.states):
            s = 0
            for j, state_j in enumerate(hmm.states):
                s += hmm.trans_probs.get(state_i, {}).get(state_j, 0) * \
                     hmm.emiss_probs.get(state_j, {}).get(observations[t+1], 0) * beta[t+1, j]
            beta[t, i] = s
    return beta

def baum_welch_update(hmm, sequences, n_iterations=10):
    # n_iterations es el número de iteraciones EM.
    for it in range(n_iterations):
        # Acumuladores para recuentos esperados
        trans_expected = defaultdict(lambda: defaultdict(float))
        emiss_expected = defaultdict(lambda: defaultdict(float))
        init_expected = defaultdict(float)
        
        total_log_likelihood = 0
        
        for observations in sequences:
            T = len(observations)
            # Calcular forward y backward para cada secuencia (de longitud T, variable)
            alpha = forward(hmm, observations)
            beta = backward(hmm, observations)
            
            # Probabilidad de la secuencia (usualmente la suma en el último tiempo)
            seq_prob = np.sum(alpha[T-1, :])
            total_log_likelihood += math.log(seq_prob + 1e-10)
            
            # Calcular gamma: probabilidad de estar en el estado i en el tiempo t
            gamma = np.zeros((T, len(hmm.states)))
            for t in range(T):
                for i in range(len(hmm.states)):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / (seq_prob + 1e-10)
            
            # Calcular xi: probabilidad de transición desde i a j entre tiempo t y t+1
            xi = np.zeros((T-1, len(hmm.states), len(hmm.states)))
            for t in range(T-1):
                norm = 0
                for i in range(len(hmm.states)):
                    for j in range(len(hmm.states)):
                        norm += alpha[t, i] * hmm.trans_probs.get(hmm.states[i], {}).get(hmm.states[j], 0) * \
                                hmm.emiss_probs.get(hmm.states[j], {}).get(observations[t+1], 0) * beta[t+1, j]
                for i in range(len(hmm.states)):
                    for j in range(len(hmm.states)):
                        xi[t, i, j] = (alpha[t, i] * hmm.trans_probs.get(hmm.states[i], {}).get(hmm.states[j], 0) * \
                                       hmm.emiss_probs.get(hmm.states[j], {}).get(observations[t+1], 0) * beta[t+1, j]) / (norm + 1e-10)
            
            # Actualizar el recuento esperado para la distribución inicial
            for i, state in enumerate(hmm.states):
                init_expected[state] += gamma[0, i]
            
            # Acumular recuento esperado para las transiciones
            for t in range(T-1):
                for i, state_i in enumerate(hmm.states):
                    for j, state_j in enumerate(hmm.states):
                        trans_expected[state_i][state_j] += xi[t, i, j]
            
            # Acumular recuento esperado para las emisiones
            for t in range(T):
                for i, state in enumerate(hmm.states):
                    # Solo se acumulan en estados emisores (si tu modelo lo requiere)
                    emiss_expected[state][observations[t]] += gamma[t, i]
        
        # Actualizar parámetros del HMM con los recuentos acumulados
        
        # Actualizar probabilidades iniciales
        new_init_probs = {}
        total_init = sum(init_expected.values())
        for state in hmm.states:
            new_init_probs[state] = init_expected[state] / (total_init + 1e-10)
        hmm.init_probs = new_init_probs
        
        # Actualizar probabilidades de transición
        new_trans_probs = {}
        for state_i in hmm.states:
            new_trans_probs[state_i] = {}
            total = sum(trans_expected[state_i].values())
            for state_j in hmm.states:
                new_trans_probs[state_i][state_j] = trans_expected[state_i][state_j] / (total + 1e-10)
        hmm.trans_probs = new_trans_probs
        
        # Actualizar probabilidades de emisión
        new_emiss_probs = {}
        for state in hmm.states:
            new_emiss_probs[state] = {}
            total = sum(emiss_expected[state].values())
            for symbol, count in emiss_expected[state].items():
                new_emiss_probs[state][symbol] = count / (total + 1e-10)
        hmm.emiss_probs = new_emiss_probs
        
        print(f"Iteración {it+1} - Log-Likelihood Total: {total_log_likelihood:.4f}")
    return hmm

# Ejemplo ilustrativo:
# Supongamos que definimos un HMM inicial con estados, probabilidades de transición, emisión e inicial
class HMM:
    def __init__(self, states, init_probs, trans_probs, emiss_probs):
        self.states = states
        self.init_probs = init_probs      # Diccionario {estado: probabilidad}
        self.trans_probs = trans_probs    # Diccionario {estado_i: {estado_j: probabilidad}}
        self.emiss_probs = emiss_probs    # Diccionario {estado: {símbolo: probabilidad}}

# Definición inicial del modelo (muy simplificada y con ejemplos arbitrarios)
states = ['M1', 'M2']  # Por ejemplo, dos estados emisores
init_probs = {'M1': 0.6, 'M2': 0.4}
trans_probs = {
    'M1': {'M1': 0.7, 'M2': 0.3},
    'M2': {'M1': 0.4, 'M2': 0.6},
}
emiss_probs = {
    'M1': {'A': 0.5, 'C': 0.5},
    'M2': {'A': 0.4, 'C': 0.6},
}
hmm = HMM(states, init_probs, trans_probs, emiss_probs)

def read_sequences(align_filename):
    alignment = []
    with open(align_filename,'r',encoding='utf-8') as align_file:
        current_line = ""
        for line in align_file:
            if line.startswith('>'):
                alignment.append(current_line)
                current_line =''
                continue
            else:
                current_line+=line
                current_line=current_line.replace('\n',"")
                current_line=current_line.replace('X',"-")
    return alignment
# Secuencias de ejemplo (cada una con longitud distinta)
sequences = read_sequences('./BioInf_ProyectoFinal/Cluster95_alineamiento.fasta')
sequences.pop(0)
# Ejecutar Baum-Welch
hmm_actualizado = baum_welch_update(hmm, sequences, n_iterations=5)