import numpy as np
import moabb
import mne
from sklearn.metrics import matthews_corrcoef



class PseudoOnlineWindow():
    """
    cria janelas deslizantes e as rotula, baseado no arigo do framework pseudo online
    os rotulos sao dados de acordo com a classe majoritaria da janela
    ==============================
    raw: objeto mne.Raw
    events: array de eventos padrao do mne
    interval: parametro do dataset que define os intervalos de imagetica
    task_ids: define quais os ids das tasks que vao ser classificadas (permite problema multiclasse ou one-versus-rest, por exemplo)
    window_size: define tamanho (em segundos) da janela
    window_step: distancia entre os inicios de duas janelas adjacentes, entao define a sobreposicao entre janelas

    *******************************arrumar para logica funcionar por amostras e nao tempo************************************
    """
    def __init__(self, raw, events, interval, task_ids, window_size, window_step):
        self.raw = raw
        self.events = events
        self.interval = interval
        self.sfreq = raw.info['sfreq']
        self.task_ids = task_ids

        self.window_size = int(window_size * self.sfreq)
        self.window_step = int(window_step * self.sfreq)
        self.t_start = int(interval[0] * self.sfreq)
        self.t_end = int(interval[1] * self.sfreq)

        self.labels = self.generate_labels()

    def generate_labels(self):
        """
        atribui uma classe para cada amostra do dado. inicializa o vetor de rotulos em 0 e atribui a classe da task
        às amostras do período de imagética
        """
        
        #aqui n_sample eh de uma forma e la embaixo de outra
        n_samples = self.raw.n_times
        labels = np.zeros(n_samples, dtype=int)

        valid_ids = list(self.task_ids.values()) #vai selecionar so os eventos que queremos

        for ev in self.events:
            ev_idx, _, ev_id = ev

            if ev_id in valid_ids:
                # considera so o periodo de imagetica para rotular como task
                start = ev_idx + self.t_start
                stop = ev_idx + self.t_end

                # garante limites do array
                start = max(0, start)
                stop = min(n_samples, stop)

                labels[start:stop] = ev_id
        return labels


    def generate_windows(self):
        """
        gera janelas para todo o dado, alem de conter a logica de desempate de classe
        return: array de dados das janelas X (shape=(2,)), array de labels y de cada janela, array de tempos em s de inicio de cada janela
        """
        X, y, times = [], [], []

        data = self.raw.get_data()  #(n_channels, n_samples)
        #n_samples ta sendo obtido de outra forma la em cima
        n_samples = data.shape[1]

        for start_idx in range (0, n_samples - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size

            window_data = data[:, start_idx : end_idx]
            window_labels = self.labels[start_idx:end_idx]

            count = np.bincount(window_labels)
            major = np.argmax(count)

            prop_major = count[major] / len(window_labels)

            if prop_major != 0.5:
                y.append(major)
            #se ha empate, vence a classe posterior; isso aqui so trata do caso de haver apenas duas classes (rest e uma task) na janela
            else:
                y.append(window_labels[-1])

            X.append(window_data)
            times.append(start_idx / self.sfreq)

        #um ponto importante a se checar é se o shape de X vai funcionar bem nos pipelines
        return np.array(X), np.array(y), np.array(times)

class PseudoOnlineEvaluation():
    """
    faz avaliacao com janelas deslizantes, tanto na mesma sessao quanto inter sessao.
    =========================================
    dataset: dataset utilizado
    pipelines: dict de pipelines
    method: pode ser 'within-session' para avaliacao na mesma sessao ou 'inter-session' para avaliacao entre sessoes
        within session: treina nas primeiras k trials definidas por ratio e testa nas demais, dentro de uma unica sessao
        inter session: treina nas prmeiras k sessoes definidas por ratio e testa nas demais sessoes
    ratio: define a proporçao dos dados usada para treino
    
    ******no geral, ainda tá meio confuso e precisa de mais robustez******
    """
    def __init__(self, dataset, pipelines, method, wsize, wstep, ratio=0.7):
        self.dataset = dataset
        self.pipelines = pipelines
        self.ratio = ratio
        self.method = method
        self.wsize = wsize
        self.wstep = wstep

        self.y_ = {}
        self.mscores = {}

    def evaluate(self):
        """
        ******ainda precisa ser validado*******
        ******adicionar verificacoes de erro e de tipos**********
        ******adicionar interpretabilidade e rastreabilidade*******      
        ******adicionar um dataframe de resultados bonitinho********
        ******falta verificar se o shape de X funciona nos pipelines********
        """

        for subject in self.dataset.subject_list:
            self.mscores[subject] = {}
            self.y_[subject] = {}
            raws_dict = {}
            raws_test = {}
            pre = self.dataset.get_data(subject=[subject])

            """
            o within session faz split nas janelas, mas talvez o melhor seja fazer split como no inter-session (separar os raws de treino e teste antes de gerar janela)
            de todo modo, isso é tranquilo de mudar
            """


            if self.method == 'within-session':
                
                #o raw mesmo fica muito aninhado dentro do dict do get_data, entao tem que acessar uns 3 dicionarios ate chegar la
                for sess, runs in pre.items():
                    raws_dict[sess] = []
                    for _, dicts in runs.items():
                        for _, data in dicts.items():
                            raws_dict[sess].append(data)
                
                for sess in pre.keys():
                    self.mscores[subject][sess] = {}
                    self.y_[subject][sess] = {}
                    
                    raw = mne.concatenate_raws(raws_dict[sess]) #concatena todos os raws e gera o split depois
                    events, event_ids = mne.events_from_annotations(raw)  #aqui da pra extrair o array de eventos pra usar no gerador de janelas

                    wgen = PseudoOnlineWindow(raw=raw,
                                              events=events,
                                              interval=self.dataset.interval,
                                              task_ids=event_ids,
                                              window_size=self.wsize,
                                              window_step=self.wstep
                                              )
                    
                    X, y, times = wgen.generate_windows()   #o times pode ser usado depois pra plot, etc

                    idx_split = int(len(X) * self.ratio)

                    X_train, y_train = X[:idx_split], y[:idx_split]
                    X_test, y_test = X[idx_split:], y[idx_split:]

                    for name, pipe in self.pipelines.items():
                        pipe.fit(X_train, y_train)
                        self.y_[subject][sess][name] = pipe.predict(X_test)

                        self.mscores[subject][sess][name] = matthews_corrcoef(y_test, self.y_[subject][sess][name])

            elif self.method == 'inter-session':
                #split de sessoes
                session_split = int(self.ratio * self.dataset.n_sessions)
                raws_dict = []
                for sess, runs in pre.items():
                    raws_test[sess] = []
                    for _, dicts in runs.items():
                        for _, data in dicts.items():
                            if sess <= session_split:
                                raws_dict.append(data)
                            else:
                                raws_test[sess].append(data)
                        
                raws_train = mne.concatenate_raws(raws_dict)

                events, event_ids = mne.events_from_annotations(raws_train)

                wgen_train = PseudoOnlineWindow(raw=raws_train,
                                                events=events,
                                                interval=self.dataset.interval,
                                                task_ids=event_ids,
                                                window_size=self.wsize,
                                                window_step=self.wstep
                                                )

                X_train, y_train, times_train = wgen_train.generate_windows()

                for name, pipe in self.pipelines.items():
                    pipe.fit(X_train, y_train)
                    for sess in range(session_split + 1, self.dataset.n_sessions + 1): #assumindo sessoes indexadas de 1 a n
                        self.mscores[subject][sess] = {}
                        self.y_[subject][sess] = {}
                        raws = mne.concatenate_raws(raws_test[sess])

                        events, event_ids = mne.events_from_annotations(raws)

                        wgen_test = PseudoOnlineWindow(raw=raws,
                                                        events=events,
                                                        interval=self.dataset.interval,
                                                        task_ids=event_ids,
                                                        window_size=self.wsize,
                                                        window_step=self.wstep
                                                        )
                        
                        X_test, y_test, times_test = wgen_test.generate_windows()

                        self.y_[subject][sess][name] = pipe.predict(X_test)

                        self.mscores[subject][sess][name] = matthews_corrcoef(y_test, self.y_[subject][sess][name])