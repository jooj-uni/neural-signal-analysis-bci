import numpy as np
import moabb
import mne

from sklearn.metrics import matthews_corrcoef
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

import time
import matplotlib.pyplot as plt
import pandas as pd

REST_LABEL = 0
REJECT_LABEL = -1

class PseudoOnlineWindow():
    """
    Segments data in windows for pseudo-online analysis.


    Parameters:
        raw: mne.Raw object
            The continuous data.
        events: arra
            MNE event array.
        interval: list
            Dataset parameter defining imagery interval.
        task_ids: dict
            Defines the tasks and its numeric IDs. It can be used to select a subset of the dataset tasks.
        window_size: float
            The window size in seconds.
        window_step: int
            Distance in seconds between the start of two consecutive windows. It can be used to set superposition between windows, when value is lower than window_size.
    
    """
    def __init__(self, raw, events, interval, task_ids, window_size, window_step, chan_list=None):
        self.raw = raw
        self.events = events
        self.interval = interval
        self.sfreq = raw.info['sfreq']
        self.task_ids = task_ids

        self.window_size = int(window_size * self.sfreq)
        self.window_step = int(window_step * self.sfreq)
        self.chan_list = chan_list

        self.t_start = int(interval[0] * self.sfreq)
        self.t_end = int(interval[1] * self.sfreq)

        self.labels = self.generate_labels()

    def generate_labels(self):
        """
        Attributes aa label for each sample. The label vector is initialized with 0 and each data point is attributed to the task label, if it is in imagery period.

        Returns:
            labels: nd array
                Label vector containing labels for each data sample.
        """
        
        n_samples = self.raw.n_times
        labels = np.zeros(n_samples, dtype=int)

        valid_ids = list(self.task_ids.values())

        for ev in self.events:
            ev_idx, _, ev_id = ev

            if ev_id in valid_ids:
                # uses only imagery period for task attribution
                start = ev_idx + self.t_start
                stop = ev_idx + self.t_end

                # ensure array limits
                start = max(0, start)
                stop = min(n_samples, stop)

                labels[start:stop] = ev_id
        return labels


    def generate_windows(self):
        """
        Generates and labels windows.
        
        Returns:
            X: nd array shape=(n_windows, n_channels, n_times)
                The windows (data).
            y: nd array
                Window labels in the same order as X.
            times: nd array
                Array of tuples. Each tuple is the timestamps (start and end) of each window. Might be useful for plotting.
        """
        X, y, times = [], [], []

        data = self.raw.get_data()
        n_samples = data.shape[1]

        for start_idx in range (0, n_samples - self.window_size, self.window_step):
            end_idx = start_idx + self.window_size

            if self.chan_list == None:
                window_data = data[:, start_idx : end_idx]
                window_labels = self.labels[start_idx:end_idx]
            else:   #channel selection
                window_data = []
                for chan in self.chan_list:
                    if chan in self.raw.ch_names:
                        window_data.append(data[chan, start_idx : end_idx])
                    else:
                        raise ValueError(f"Channel {chan} is not in {self.raw.ch_names}")
                window_labels = self.labels[start_idx:end_idx]

            count = np.bincount(window_labels)
            major = np.argmax(count)

            prop_major = count[major] / len(window_labels)

            # class draw proportion
            n_classes = len(np.unique(window_labels))
            draw_prop = 1 / n_classes

            # in case of draw, the posterior label wins
            if prop_major != draw_prop:
                y.append(major)
            else:
                y.append(window_labels[-1])

            X.append(window_data)
            times.append(((start_idx / self.sfreq), (end_idx / self.sfreq)))

        return np.array(X), np.array(y), np.array(times)


class IdleBaseline(BaseEstimator, TransformerMixin):
    """
    Applies baseline correction. It uses a fixed baseline. This transformer has to be applied to windowed data.

    Parameters:
        rest_label: int
            Label representing idle state.
    
    Returns:
        X: nd array shape=(n_windows, n_channels, n_times)
            Baseline corrected windowed data.
    """

    def __init__(self, rest_label=0):
        self.rest_label = 0

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Missing labels array")
        
        idle_windows = (y == self.rest_label)

        if not np.any(idle_windows):
            raise ValueError("There are no rest windows")
        
        self.baseline_ = X[idle_windows].mean(axis=0)
        return self

    def transform(self, X):
        return np.subtract(X, self.baseline_)


class PSD(BaseEstimator, TransformerMixin):

    def __init__(self, sfreq, fmin=0, fmax=None):
        self.sfreq = sfreq
        self.fmin = fmin
        self.fmax = fmax

    def fit (self, X, y=None):
        return self

    def transform(self, X):
        
        fmax = self.fmax if self.fmax is not None else self.sfreq / 2

        psds, _ = mne.time_frequency.psd_array_welch(X, 
                                                     sfreq=self.sfreq,
                                                     fmin=self.fmin,
                                                     fmax=self.fmax,
                                                     average='mean')
        
        return psds
        
class IdleDetection():
    """
    Classifier for idle state detection.
    """
    def __init__(self, model, threshold=0.6):
        self.threshold = threshold
        self.model = model

    def fit(self, X, y):
        y_idle = (y != REST_LABEL).astype(int)
        self.model.fit(X, y_idle)
        return self

    def is_idle(self, window):
        p_idle = self.model.predict_proba(window)[0, 0]

        return 0 if (p_idle < self.threshold) else 1, p_idle


class PseudoOnlineEvaluation():
    """
    Evaluates windowed data in pseudo-online manner (simulating real time asynchronous BCI). It can be done within-session, evaluating only one session data, or inter-session, evaluating performance across sessions, without
    breaking signal causality.

    Parameters:
        dataset: MOABB dataset
            Dataset to be used. It has to be a MOABB dataset object.
        pipelines: dict
            Sklearn pipelines dictionary, as used in MOABB.
        method: string
            Either 'within-session' or 'inter-session'.
                within-session: trains models on first k windows, testing on the remaining ones.
                inter-session: trains models on first k sessions, testing on the remaining ones; doesn't violate data causality.
        wsize: float
            Window size.
        wstep: int
            Distance between consecutive windows (start to start).
        subjects: list
            List of subjects to process.
        ratio: float
            Proportion of data to be used in training.
        threshold: float
            Confidence threshold for classification.
        
    """
    def __init__(self, dataset, class_pipelines, method, wsize, wstep, subjects, feature_pipeline=None, ratio=0.7, threshold=0.6, no_run=False):
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.class_pipelines = class_pipelines
        self.ratio = ratio
        self.method = method
        self.wsize = wsize
        self.wstep = wstep
        self.subjects = subjects
        self.task_threshold = threshold
        self.no_run = no_run
        
        self.results_ = []
        self.model_results_ = []

    def raw_concat(self, raw_list):
        """
        Auxiliary function for raw data concat.
        """
        if len(raw_list) == 0:
            raise ValueError("Raw list is empty.")
        elif len(raw_list) == 1:    #aqui, o raw de uma sessao pode ser constituido de 1 ou mais runs, por isso essa verificação
            if type(raw_list[0]) != list:
                return raw_list[0]
            else:
                return mne.concatenate_raws(raw_list[0])
        else:
            return mne.concatenate_raws(raw_list)
        
    def window_process(self, subject, sess, X_test, y_test, times_test):
        # window processing loop
        for window in range(len(X_test)):
            window_start = times_test[window][0]
            window_end = times_test[window][1]

            t_start = time.perf_counter()

            if (self.feature_pipeline != None):
                feature_window = self.feature_pipeline.transform([X_test[window]])
            else:
                feature_window = [X_test[window]]

            t_end = time.perf_counter()

            t_transform = t_end - t_start

            t_start = time.perf_counter()
            
            idle_window, idle_proba = self.idle_detector.is_idle(feature_window)

            t_end = time.perf_counter()
            t_idle_detect = t_end - t_start

            task_proba = None

            if not idle_window:
                for name, pipe in self.class_pipelines:
                    t_start = time.perf_counter()
                    
                    probs = pipe.predict_proba(feature_window)[0]

                    task_proba = np.max(probs)

                    if task_proba < self.task_threshold:
                        y_pred = REJECT_LABEL
                    else:
                        y_pred = probs.argmax()

                    t_end = time.perf_counter()

                    t_task_predict = t_end - t_start

                    correct = (y_pred == y_test[window])

                    res = {
                        "dataset": self.dataset,
                        "subject": subject,
                        "session": sess,
                        "method": self.method,
                        "pipeline": name,
                        "window": window,
                        "window_start": window_start,
                        "window_end": window_end,
                        "is_idle": idle_window,
                        "t_transform": t_transform,
                        "t_idle_detect": t_idle_detect,
                        "t_task_predict": t_task_predict,
                        "t_predict": (t_transform + t_idle_detect + t_task_predict),
                        "y_pred": y_pred,
                        "idle_proba": idle_proba,
                        "task_proba": task_proba,
                        "y_true": y_test[window],
                        "correct": correct
                    }

                    self.results_.append(res)
            else:
                y_pred = REST_LABEL
                t_task_predict = 0
                correct = (y_pred == y_test[window])

                res = {
                    "dataset": self.dataset,
                    "subject": subject,
                    "session": sess,
                    "method": self.method,
                    "pipeline": "idle",
                    "window": window,
                    "window_start": window_start,
                    "window_end": window_end,
                    "is_idle": idle_window,
                    "t_transform": t_transform,
                    "t_idle_detect": t_idle_detect,
                    "t_task_predict": t_task_predict,
                    "t_predict": (t_transform + t_idle_detect + t_task_predict),
                    "y_pred": y_pred,
                    "idle_proba": idle_proba,
                    "task_proba": task_proba,
                    "y_true": y_test[window],
                    "correct": correct
                }

                self.results_.append(res)

    def evaluate(self):
        """
        Main function for processing data.
        """

        self.idle_detector = IdleDetection()

        for subject in self.subjects:
            if subject not in self.dataset.subject_list:
                raise ValueError(f"Invalid subject index: {subject}")
            else:
                print(f"Processing subject {subject}...")

                raws_dict = {}
                raws_test = {}
                pre = self.dataset.get_data(subjects=[subject])
                
                session_keys = []   # stores session ids (not always int)

                if self.method == 'within-session':
                    
                    # raw extraction from moabb dataset
                    for _, runs in pre.items():
                        for sess, dicts in runs.items():
                            session_keys.append(sess)
                            raws_dict[sess] = []
                            for _, data in dicts.items():
                                raws_dict[sess].append(data)
                    
                    for sess in session_keys:
                        print(f"Processing session {sess} subject {subject}...")
                        raw = self.raw_concat(raws_dict[sess])
                        events, event_ids = mne.events_from_annotations(raw)

                        wgen = PseudoOnlineWindow(raw=raw,
                                                events=events,
                                                interval=self.dataset.interval,
                                                task_ids=event_ids,
                                                window_size=self.wsize,
                                                window_step=self.wstep
                                                )
                        
                        X, y, times = wgen.generate_windows()

                        idx_split = int(len(X) * self.ratio)

                        times_test = times[idx_split:]

                        X_train, y_train = X[:idx_split], y[:idx_split]
                        X_test, y_test = X[idx_split:], y[idx_split:]

                        if (self.no_run):
                            return X_train, y_train, X_test, y_test
                        
                        # feature pipeline training
                        t_start = time.perf_counter()
                        self.feature_pipeline.fit(X_train, y_train)
                        t_end = time.perf_counter()

                        t_feature_train = t_end - t_start

                        res = {
                            "pipeline": "feature",
                            "method": self.method,
                            "t_train": t_feature_train
                                
                        }

                        self.model_results_.append(res)
                        
                        X_train = self.feature_pipeline.transform(X_train)
                        
                        # idle detector training
                        t_start = time.perf_counter()
                        self.idle_detector.fit(X_train, y_train)
                        t_end = time.perf_counter()

                        t_idle_train = t_end - t_start

                        res = {
                            "pipeline": "idle",
                            "method": self.method,
                            "t_train": t_idle_train
                            
                        }

                        self.model_results_.append(res)

                        # mask for task windows
                        mask = y_train != REST_LABEL
                        X_task = X_train[mask]
                        y_task = y_train[mask]
                        
                        # task classifier training
                        for name, model in self.class_pipelines:
                            print("Fitting task classifier...")
                            t_start = time.perf_counter()
                            model.fit(X_task, y_task)
                            t_end = time.perf_counter()

                            t_train = t_end - t_start
                            print("Done fitting!")

                            res = {
                                "pipeline": name,
                                "method": self.method,
                                "t_train": t_train
                                
                            }

                            self.model_results_.append(res)

                        self.window_process(subject, sess, X_test, y_test, times_test)
            
            
                elif self.method == 'inter-session':
                    if self.dataset.n_sessions > 1:
                        session_split = int(self.ratio * self.dataset.n_sessions)
                        raws_list = []
                        raws_train = []

                        print(f"Splitting index is {session_split}, dataset has {self.dataset.n_sessions} sessions per subject.")

                        for _, runs in pre.items():
                            for sess, dicts in runs.items():
                                session_keys.append(sess)
                                raws_test[sess] = []
                                raws_dict[sess] = []
                                for _, data in dicts.items():
                                    raws_dict[sess].append(data)
                        
                        #essa verificação é porque eu acahva que o int() arredondava pra cima o valor... de todo jeito, nao faz mal deixar isso aqui
                        if session_split == self.dataset.n_sessions:
                            train_sessions = session_keys[:(session_split - 1)]
                            test_sessions = session_keys[(session_split - 1):]
                            for sess, data in raws_dict.items():
                                if (session_keys.index(sess) + 1) < session_split:
                                    raws_list.append(data)
                                else:
                                    raws_test[sess].append(data)
                        else:
                            train_sessions = session_keys[:session_split]
                            test_sessions = session_keys[session_split:]
                            for sess, data in raws_dict.items():
                                if (session_keys.index(sess) + 1) <= session_split:
                                    raws_list.append(data)
                                else:
                                    raws_test[sess].append(data)
                        
                        
                        raws_train = self.raw_concat(raws_list)

                        events, event_ids = mne.events_from_annotations(raws_train)

                        wgen_train = PseudoOnlineWindow(raw=raws_train,
                                                            events=events,
                                                            interval=self.dataset.interval,
                                                            task_ids=event_ids,
                                                            window_size=self.wsize,
                                                            window_step=self.wstep
                                                            )

                        X_train, y_train, times_train = wgen_train.generate_windows()

                        if (self.no_run):
                            return X_train, y_train, X_test, y_test
                        
                        # feature pipeline training
                        t_start = time.perf_counter()
                        self.feature_pipeline.fit(X_train, y_train)
                        t_end = time.perf_counter()

                        t_feature_train = t_end - t_start

                        res = {
                            "pipeline": "feature",
                            "method": self.method,
                            "t_train": t_feature_train
                                
                        }

                        self.model_results_.append(res)
                        
                        X_train = self.feature_pipeline.transform(X_train)
                        
                        # idle detector training
                        t_start = time.perf_counter()
                        self.idle_detector.fit(X_train, y_train)
                        t_end = time.perf_counter()

                        t_idle_train = t_end - t_start

                        res = {
                            "pipeline": "idle",
                            "method": self.method,
                            "t_train": t_idle_train
                            
                        }

                        self.model_results_.append(res)

                        # mask for task windows
                        mask = y_train != REST_LABEL
                        X_task = X_train[mask]
                        y_task = y_train[mask]

                        # task classifier training
                        for name, model in self.class_pipelines:
                            print("Fitting task classifier...")
                            t_start = time.perf_counter()
                            model.fit(X_task, y_task)
                            t_end = time.perf_counter()

                            t_train = t_end - t_start
                            print("Done fitting!")

                            res = {
                                "pipeline": name,
                                "method": self.method,
                                "t_train": t_train
                                
                            }

                            self.model_results_.append(res)
                            
                        for sess in test_sessions:
                            print(f"Testing in session {sess}...")

                            raws = self.raw_concat(raws_test[sess])

                            events, event_ids = mne.events_from_annotations(raws)

                            wgen_test = PseudoOnlineWindow(raw=raws,
                                                                events=events,
                                                                interval=self.dataset.interval,
                                                                task_ids=event_ids,
                                                                window_size=self.wsize,
                                                                window_step=self.wstep
                                                                )
                                
                            X_test, y_test, times_test = wgen_test.generate_windows()

                            self.window_process(subject, sess, X_test, y_test, times_test)
                    else:
                        raise ValueError("There are not enough sessions for evaluation.")
                    
            if len(self.results_):
                self.results_ = pd.DataFrame(self.results_)
                self.results_.to_csv(f"results-S{subject}.csv", index=False)
            if len(self.model_results_):
                self.model_results_ = pd.DataFrame(self.model_results_)
                self.model_results_.to_csv(f"model-results-S{subject}.csv", index=False)



            
                                   # mcc_acc_sess = matthews_corrcoef(y_test[:window+1], predictions_sess[:window+1])    #score acumulado dentro da sessão
                                   # mcc_acc = matthews_corrcoef(y_all, predictions)   #score acumulado entre sessões