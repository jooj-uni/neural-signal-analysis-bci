import numpy as np
import moabb
import mne

class PseudoOnlineWindow():
    """
    a classe cria as janelas e as rotula baseado na proporção de imagética e não imagética
    os dados Raw devem ser extraídos do dataset com get_data() antes
    os outros parametros do init podem ser extraídos do objeto dataset também, a nao ser os params de window, que são definidos a gosto

    um ponto importante é que ainda não tá muito claro como extrair os eventos do dataset, na classe levei em conta o array de eventos, típico do mne, mas não sei
    como extrair isso no moabb.

    acho que deve ter um jeito mais elegante de escrever esse codigo, mas por enquanto...

    *******************************arrumar para logica funcionar por amostras e nao tempo************************************
    """
    def __init__(self, raw, events, interval, task_ids, window_size, window_step):
        self.raw = raw
        self.events = events
        self.interval = interval
        self.sfreq = raw.info['sfreq']
        self.task_ids = task_ids
        self.window_size = window_size
        self.window_step = window_step

        #extrai tempos de começo e fim da imagetica (considerando um o periodo fixo de imagetica determinado no interval)
        self.task_times = [
            (ev[0] / self.sfreq + interval[0],
             ev[0] / self.sfreq + interval[1],
             ev[2])
            for ev in events if ev[2] in task_ids
        ]
        
        self.task_intervals = [(np.arange(t0, t1), ev) for t0, t1, ev in self.task_times]

    def label_window(self, start, stop):
        """
        atribui os labels para cada janela
            0: nothing -> se a maior parte nao representa imagetica
            1: imagetica -> se a maior parte representa imagetica
        se ambos os rotulos (nothing ou task_id) tiverem mesma proporcao, atribui para a janela o rotulo mais posterior 

        a principio to considerando que vai acontecer de duas uma: o sujeito nao tenta controlar ou o sujeito tenta controlar, mas nao to considerando qual a imagetica que ele ta tentando performar, apenas se teve uma task (qualquer que seja)

        ***** ainda da pra otimizar
        ***** essa funcao leva em conta que uma janela vai conter no maximo uma task; precisa adaptar se for considerar janelas grandes o suficiente para abarcar mais de 1 task

        """

        
        window_times = np.arange(start, stop)#***********arrumar para calcular em amostras

        nothing_qt = [p for p in window_times if not any(p in q for q, _ in self.task_intervals)]
        nothing_prop = len(nothing_qt)/self.window_size


        if nothing_prop > 0.5:
            return 0
        elif nothing_prop < 0.5:
            for p in window_times:
                for q, ev in self.task_intervals:
                    if p in q: return ev
        else:
            for t0, _, ev_id in self.task_times:
                return ev_id if t0 in window_times else 0 #se t0 ta em window_times, a task é a classe posterior; se t1 ta em window_times, nothing é posterior


    def generate_window(self):
        X, Y, times = [], [], []

        """
        loop para pegar os indices de inicios das janelas
        o arange gera de 0 (inicio dos dados) ate tempo total (ultimo dado) - uma janela, para evitar acesso inválido
        """
        for start in np.arange(0, self.raw.times[-1] - self.window_size, self.window_step):
            stop = start + self.window_size

            data = self.raw.get_data(
                start=int(start * self.sfreq),
                stop=int(stop * self.sfreq)
            )

            label = self.label_window(start, stop)
            X.append(data)
            Y.append(label)
            times.append((start, stop))
        return np.array(X), np.array(Y), np.array(times)
    