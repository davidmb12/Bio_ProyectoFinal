import math
class HMM:
    def __init__(self,states,start_state,end_state,trans_probs,emiss_probs):
        self.states= states
        self.start_state = start_state
        self.end_state = end_state
        self.trans_probs = trans_probs
        self.emiss_probs = emiss_probs
    def viterbi(self,observations):
        """
        Implementacion del algoritmo de Viterbi usando log-probabilidades para mayor estabilidad.
        observations: lista o cadena de observaciones (simbolos de aminoacidos)
        Retorna la secuencia de estados mas probable y el log de la probabilidad de esa trayectoria
        """
        
        T= len(observations)
        # dp[t] sera un diccionario que contiene la log-probabilidad maxima para cada estado al tiempo t
        # bp[t] contendra el backpointer para reconstruir la trayectoria
        dp =[{} for _ in range(T+1)]
        bp =[{} for _ in range(T+1)]
        
        # Inicializacion: desde el estado S
        dp[0][self.start_state] = 0
        # Para t = 1 a T, iterar sobre cada observacion
        for t in range(1,T+1):
            o = observations[t-1]
            for curr in self.states:
                # Solo consideramos estados emisores: ignorar S y E
                if curr in [self.start_state, self.end_state]:
                    continue
                max_log_prob = -math.inf
                best_prev = None
                # Revisamos todas las posibles transiciones provenientes de los estados con los que terminamos en el paso anterior
                for prev, log_prob_prev in dp[t-1].items():
                    # Verificamos que exista una transicion de prev a curr
                    if prev in self.trans_probs and curr in self.trans_probs[prev]:
                        # Obtenemos el log de la probabilidad de transición
                        log_trans= math.log(self.trans_probs[prev][curr])
                        # Obtenemos el log de la probabilidad de emisión en el estado curr para la observación actual
                        # Si la emision no esta definida, asignamos un valor pequeño para evitar log(0)
                        emiss_p =self.emiss_probs.get(curr,{}).get(o,1e-10)
                        log_emit = math.log(emiss_p)
                        candidate = log_prob_prev+log_trans+log_emit
                        if candidate > max_log_prob:
                            max_log_prob = candidate
                            best_prev = prev
                # Solo agregamos el estado si se encontro al menos un camino
                if best_prev is not None:
                    dp[t][curr] = max_log_prob
                    bp[t][curr] = best_prev
                    
        # Paso de terminacion: Transitar desde alguno de los estados emisores al estado final
        max_log_prob = -math.inf
        best_last_state = None
        for state,log_prob in dp[T].items():
            if state in self.trans_probs and self.end_state in self.trans_probs[state]:
                log_trans=math.log(self.trans_probs[state][self.end_state])
                candidate = log_prob+log_trans
                if candidate > max_log_prob:
                    max_log_prob = candidate
                    best_last_state = state
        # Si no se encuentra camino valido, retornamos un mesaje de error
        if best_last_state is None:
            raise ValueError("No se encontro una trayectoria valida hacia el estado final")

        # Backtracking para reconstruir la trayectoria
        path = [self.end_state]
        curr_state = best_last_state
        for t in range(T,0,-1):
            path.append(curr_state)
            curr_state = bp[t][curr_state]
        path.append(self.start_state)
        path = list(reversed(path))
        return path,max_log_prob
