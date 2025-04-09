from Bio.Align.Applications import ClustalOmegaCommandline
from Bio import AlignIO
from Bio import SeqIO
import os
import subprocess
from collections import defaultdict
import pprint

from hmm_implementation import HMM
from baum_welch import baum_welch_update

print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))
def extract_cluster_sequences(cluster_filename,fasta_filename,output_filename):
    with open(cluster_filename,'r',encoding='utf-8') as cluster_file:
        cluster_ids = []
        for line in cluster_file:
            if line.startswith('>'):
                continue
            if(line.__contains__("*")):
                seq_id = line.split('>')[1].split('...')[0]
            cluster_ids.append(seq_id)
    with open(fasta_filename, 'r', encoding='utf-8') as fasta_file:
        sequences = SeqIO.parse(fasta_file, 'fasta')
        filtered_seqs = [seq for seq in sequences if seq.id in cluster_ids]
    with open(output_filename, 'w', encoding='utf-8') as output:
        SeqIO.write(filtered_seqs, output, 'fasta')
# 2. Identificar las columnas "match" basadas en el umbral de no-gap (>=50%)
extract_cluster_sequences(
    cluster_filename='./BioInf_ProyectoFinal/conoserver_250206_protein_cdhit.fa.clstr',
    fasta_filename='./BioInf_ProyectoFinal/conoserver_250206_protein.fa',
    output_filename='./BioInf_ProyectoFinal/cluster_sequences.fasta'
)   
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
        
           

def identify_match_columns(alignment, threshold=0.5):
    nseq = len(alignment)
    ncol = len(alignment[1])
    match_cols =[]
    for j in range(ncol):
        count_non_gap = sum(1 for seq in alignment if seq[j] != '-')
        if count_non_gap / nseq >= threshold:
            match_cols.append(j)
    return match_cols

# Construir la secuencia de estados para cada secuencia
#     - Para cada columna match: si no es gap -> estado M; Si lo es -> estado D.
#     - Para cada columna que no es match: si hay residuo -> estado I asociado al ultimo match.
#     - Se incluyen estados de inicio 'S' y fin 'E'

def get_state_path(seq,match_cols):
    state_path =['S']
    match_index = 0
    last_match_index = None
    ncol = len(seq)
    for j in range(ncol):
        if j in match_cols:
            match_index +=1
            if seq[j] =='-':
                state_path.append(f'D{match_index}')
            else:
                state_path.append(f'M{match_index}')
            last_match_index = match_index
        else:
            # Columna de insercion
            if seq[j] !='-':
                # El estado de incersion se asocia al ultimo match (o 0 si no hay match previo)
                ins_state = f'I{last_match_index if last_match_index is not None else 0}'
                state_path.append(ins_state)
    state_path.append('E')
    return state_path

# Calculo de las transiciones entre estados
def calculate_transitions(state_paths):
    transition_counts = defaultdict(int)
    for path in state_paths:
        for i in range(len(path)-1):
            transition = (path[i],path[i+1])
            transition_counts[transition]+=1
    return transition_counts

# Calculo de la matriz de emisiones
def get_emissions(seq, match_cols):
    emissions = []
    match_index = 0 
    last_match_index = None
    ncol = len(seq)
    for j in range(ncol):
        if j in match_cols:
            match_index +=1
            if seq[j] != '-':
                state = f'M{match_index}'
                emissions.append((state,seq[j]))
            # Si es gap, se considera delecion (no emite)
            last_match_index = match_index
        else:
            # Columna de insercion: si hay residuo, emite
            if seq[j] !='-':
                ins_state=f'I{last_match_index if last_match_index is not None else 0}'
                emissions.append((ins_state,seq[j]))
    return emissions


input_fasta = './BioInf_ProyectoFinal/cluster_sequences.fasta'
output_fasta = './BioInf_ProyectoFinal/aligned_cluster_sequences_01.fasta'

cmd = ['mafft.bat', '--anysymbol','--localpair', input_fasta]
result = subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')

if result.returncode != 0:
    print("MAFFT Error:", result.stderr)
else:
    with open(output_fasta, 'w', encoding='utf-8') as aligned_output:
        aligned_output.write(result.stdout)

alignment = AlignIO.read(output_fasta, 'fasta')
print(alignment)
alignment =read_sequences('./BioInf_ProyectoFinal/aligned_cluster_sequences_01.fasta')
alignment.pop(0)

match_cols = identify_match_columns(alignment, threshold=0.5)
# Generar la trayectoria de estados para cada secuencia
state_paths = [get_state_path(seq,match_cols) for seq in alignment]
for i, path in enumerate(state_paths):
    print(f'Secuencia {i+1} - trayectoria de estados: {path}')

# Generar las transiciones entre estados
transition_counts = calculate_transitions(state_paths)
for trans,count in transition_counts.items():
    print(f"{trans}: {count}")

# Generar matriz de emisiones
emission_counts = defaultdict(lambda:defaultdict(int))
for seq in alignment:
    for state, symbol in get_emissions(seq,match_cols):
        emission_counts[state][symbol]+=1

print("\nContabilización de Emisiones (por estado):")
for state, counts in emission_counts.items():
    print(f"{state}: {dict(counts)}")
    
# Calcular las probabilidades de emision y transicion (usando pseudoconteos)
alpha= 1
protein_alphabet = list("ARNDCEQGHILKMFPSTWYVUOBZXJ")
emission_probabilities = {}
for state,counts in emission_counts.items():
    total = sum(counts.values())+ alpha * len(protein_alphabet)
    emission_probabilities[state]={}
    for letter in protein_alphabet:
        emission_probabilities[state][letter] = (counts.get(letter,0)+alpha)/total
        
print("\nProbabilidades de Emisión (con pseudocontadores):")
pprint.pprint(emission_probabilities)

alpha_trans = 1  # Pseudocontador para transiciones
from_state_transitions = defaultdict(lambda:defaultdict(int))
for(from_state,to_state), count in transition_counts.items():
    from_state_transitions[from_state][to_state] =count

transition_probabilities = {}
for from_state,transitions in from_state_transitions.items():
    # Numero de transiciones posibles observadas desde 'from_state'
    total = sum(transitions.values())+alpha_trans *len(transitions)
    transition_probabilities[from_state]={}
    for to_state,count in transitions.items():
        transition_probabilities[from_state][to_state] = (count+alpha_trans)/total

print("\nProbabilidades de Transición (con pseudocontadores):")
pprint.pprint(dict(transition_probabilities))

def plot_transition_graph(trans_probs):
    """
    Dibuja un grafo dirigido de las transiciones del HMM usando networkx y matplotlib.
    
    Parámetros:
      trans_probs: diccionario de probabilidades de transición. Ejemplo:
          {
              "S": {"M1": 0.6, "I0": 0.4},
              "M1": {"M2": 0.7, "I0": 0.2, "E": 0.1},
              "I0": {"M1": 0.5, "I0": 0.3, "E": 0.2},
              "M2": {"E": 1.0},
              "E": {}
          }
    """
    import networkx as nx
    import matplotlib.pyplot as plt

    # Crear un grafo dirigido
    G = nx.DiGraph()
    
    # Agregar nodos y aristas al grafo utilizando el diccionario de transiciones
    for state_from, transitions in trans_probs.items():
        for state_to, prob in transitions.items():
            # Se redondea la probabilidad para mejorar la visualización
            G.add_edge(state_from, state_to, weight=round(prob, 2))
    
    # Definir la posición de los nodos 
    pos = nx.spring_layout(G, seed=42)
    
    # Crear la figura
    plt.figure(figsize=(8, 6))
    
    # Dibujar el grafo: nodos y aristas
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500,
            arrowsize=20, font_size=10, edge_color='gray')
    
    # Extraer y dibujar las etiquetas de las aristas (probabilidades)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Grafo de Transiciones del HMM")
    plt.axis('off')
    plt.show()

# Llamamos a la función para visualizar el grafo
# plot_transition_graph(emission_probabilities)
emitter_states = list(emission_probabilities.keys())
all_states = ['S'] +emitter_states + ['E']

# Agregar las transiciones desde el estado S
if 'S' not in transition_probabilities:
    transition_probabilities['S'] = {'M1':1.0}



# Imprimimos para revisar
print("\nEstados del Modelo:", all_states)
print("\nTransiciones Finales del Modelo:")
pprint.pprint(dict(transition_probabilities))
print("\nEmisiones Finales del Modelo:")
pprint.pprint(emission_probabilities)

# Inicializamos el HMM
hmm_model = HMM(
    states=all_states,
    start_state='S',
    end_state='E',
    trans_probs=transition_probabilities,
    emiss_probs=emission_probabilities
)

sequences = read_sequences('./BioInf_ProyectoFinal/Cluster95_alineamiento.fasta')
sequences.pop(0)
print(hmm_model.start_state)
print(hmm_model.emiss_probs)
print(hmm_model.trans_probs)
# hmm_actualizado = baum_welch_update(hmm_model, sequences, n_iterations=5)
# Suponiendo que queremos decodificar una secuencia de observaciones (por ejemplo, una cadena de aminoácidos)
# Nota: La secuencia de observación debe corresponder a los símbolos (aminoácidos) y se decodificará en estados emisores.
observations = "MGMRMMFTVFLLVVLATTVVSFPSDRASDGRNAAANDKASDVITLALKGCCSNPVCHLEHSNLCGRRR"  # Por ejemplo
print("\nSecuencia de Observación:", observations)

try:
    best_path, best_log_prob = hmm_model.viterbi(list(observations))
    print("\nTrayectoria Viterbi más probable:")
    print(" -> ".join(best_path))
    print("Log-probabilidad de la trayectoria:", best_log_prob)
except ValueError as ve:
    print("Error al ejecutar Viterbi:", ve)