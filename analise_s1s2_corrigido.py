#%%
import mne
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   mne                           import Epochs, pick_types
from autoreject import AutoReject
from scipy.stats import zscore
from mne_icalabel import label_components
from mne.preprocessing import ICA


sujeitos = [1, 2]
dados = {}
raws = {}
noise_chan = {}

for suj in sujeitos:
    path_data         = 'C:/Users/aihon/Desktop/proc_EEG/'
    sess              = 2
    filename          = f'S{suj}_Session_{sess}.mat'
    ind_names         = ['data', 'time', 'positionx', 'positiony', 'SRATE', 'TrialData', 'metadata', 'chaninfo']
    trial_data_labels = ['tasknumber', 'runnumber', 'trialnumber', 'targetnumber', 'triallength',
                         'targethitnumber', 'resultind', 'result', 'forcedresult', 'artifact']
    feedback_len_s    = 3.5
    trial_len_s       = 7.5
    
    data              = sio.loadmat(path_data + filename)
    BCI_data          = data['BCI'][0][0]
    sampling_freq     = BCI_data[ind_names.index('SRATE')][0][0]
    raw_trials        = BCI_data[ind_names.index('TrialData')][0]
    channels_info     = BCI_data[ind_names.index('chaninfo')][0][0]
    noise_chan[suj]   = BCI_data[ind_names.index('chaninfo')][0][0][1]
    
    # Nomes e posições dos canais
    ch_names          = [str(c[0]) for c in channels_info[0][0]]
    positions         = np.array([[c[1][0][0], c[2][0][0], c[3][0][0]] for c in channels_info[2][0]])
    montage           = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, positions)), coord_frame='head')
    info              = mne.create_info(ch_names=ch_names, sfreq=sampling_freq, ch_types='eeg')
    info.set_montage(montage)
    
    # Processamento dos dados dos trials
    df                = pd.DataFrame(raw_trials, columns=trial_data_labels).dropna()
    for col in trial_data_labels:
      df[col]         = df[col].apply(lambda x: x[0][0])
    df_clean          = df[(df['triallength'] >= feedback_len_s) & (~df['targetnumber'].isin([3, 4]))]
    
    # Construção do sinal contínuo
    trial_samples     = int(trial_len_s * sampling_freq)
    all_data          = [BCI_data[ind_names.index('data')][0][int(r['trialnumber']) - 1][:, :trial_samples] for _, r in df_clean.iterrows()]
    epochs_data       = np.concatenate(all_data, axis=1)
    epochs_data = epochs_data * 1e-6
    
    # Criação do RawArray e pré-processamento
    raw               = mne.io.RawArray(epochs_data, info)
    raw.set_eeg_reference(ref_channels='average')
    
    raws[suj] = raw
    
    # EPOCAGEM
    # Eventos e criação dos Epochs
    target_labels    = df_clean['targetnumber'].astype(int).to_numpy()
    
    cue_idx = 2001
    trial_starts = np.arange(0, len(raw.times), trial_samples)
    events_idx = trial_starts + cue_idx

    events           = np.column_stack((events_idx, np.zeros(len(target_labels), dtype=int), target_labels))
    event_id         = dict(right=1, left=2)
    picks            = pick_types(raw.info, eeg=True)
    epochs           = Epochs(raw, events.astype(int), event_id=event_id, picks=picks, preload=True, baseline=(0, 0), tmin=-4, tmax=3.5)
    
    epochs.filter(0.5, None , fir_design='firwin', skip_by_annotation='edge')
    epochs.notch_filter(freqs=60)
    
    
    dados[suj] = epochs
#%%
for suj in sujeitos:
    print(f"sujeito {suj}:", noise_chan[suj])
    
#%% rejeição de epochs
# no mne, da pra fazer isso tanto na criacao dos epochs quando depois, usando amplitude peak to peak
# eh so definir um dicionario que vai dar o valor de threshold nos canais de um determinado tipo e, se qualquer canal ultrapassar, ele rejeita a epoch

reject_criteria = dict(
    eeg = 72*10e-6
    )

copia = {}

for suj in sujeitos:
    copia[suj] = dados[suj].copy()
    # o param reject usa o threshold como limite maximo e o param flat usa como limite minimo
    copia[suj].drop_bad(reject=reject_criteria)
    print(copia[suj].drop_log)
    copia[suj].plot()

# o drop_bad tbm aceita como param uma funcao, pode ser lambda ou uma funcao definida separadamente

#%% PIPELINE PARA REJEICAO DE EPOCHS BASEADO EM FREQUENCIA
# a ideia eh computar a potencia espectral na banda de interesse e pegar descartar trials cujo psd tem pouca correlacao com a media

dados_limpos = {}

def rejeicao_por_freq(epochs, fmin, fmax, limiar):
    
    n_epochs, n_channels, n_times = epochs.get_data().shape
    
    psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    psd_data = psd.get_data()   #(n_epochs, n_channels, n_freqs)
    
    corrs = np.zeros((n_epochs, n_channels))
    
    for i in range(n_epochs):
        # media das trials sem considerar a trial atual
        media_outros = np.mean(np.delete(psd_data, i, axis=0), axis=0)
        
        # correlação por canal
        for ch in range(n_channels):
            corrs[i, ch] = np.corrcoef(psd_data[i, ch], media_outros[ch])[0, 1]
            
    # Z-score das correlações
    zscores = zscore(corrs, axis=0)
    
    mascara_ruins = np.zeros(n_epochs, dtype=bool)
    
    # mascara de rejeição
    for i in range(n_epochs):
        n_ruins = np.sum(zscores[i] < limiar)
        if n_ruins >= 3:
            mascara_ruins[i] = True
    
    print(f"Trials rejeitados: {np.sum(mascara_ruins)} / {n_epochs}")
    
    return epochs[~mascara_ruins]

for suj in sujeitos:
    dados_limpos[suj]= rejeicao_por_freq(dados[suj], fmin=8, fmax=30, limiar=-2)
    dados_limpos[suj].plot()
        



#FIM DO PIPELINE

#%% PIPELINE DE REJEICAO DE TRIALS BASEADO EM CORRELACAO
#descarta trials em que a correlacao do sinal no tempo eh baixa em relacao a media em ao menos 3 canais
#nao funcionou bem

dados_limpos = {}

def rejeicao_por_correlacao(epochs, limiar):
    
    
    "pega o zscore da correlação de uma trial com a média de todas as outras e rejeita se for outlier"""
    
    
    dados = epochs.get_data()
    n_epochs, n_channels, n_times = dados.shape
    
    corrs = np.zeros((n_epochs, n_channels))
    
    for i in range(n_epochs):
        # media das trials sem considerar a trial atual
        media_outros = np.mean(np.delete(dados, i, axis=0), axis=0)
        
        # correlação por canal
        for ch in range(n_channels):
            corrs[i, ch] = np.corrcoef(dados[i, ch], media_outros[ch])[0, 1]
            
    # Z-score das correlações
    zscores = zscore(corrs, axis=0)
    
    mascara_ruins = np.zeros(n_epochs, dtype=bool)
    
    # mascara de rejeição
    for i in range(n_epochs):
        n_ruins = np.sum(zscores[i] < limiar)
        if n_ruins >= 3:
            mascara_ruins[i] = True
    
    print(f"Trials rejeitados: {np.sum(mascara_ruins)} / {n_epochs}")
    
    return epochs[~mascara_ruins]

for suj in sujeitos:
    dados_limpos[suj]= rejeicao_por_correlacao(dados[suj], limiar=-2)
    dados_limpos[suj].plot()

#%% PIPELINE DE REJEICAO AUTOMATICA DE TRIALS USANDO AUTOREJECT
#faz umas conta ai pra ver qual o melhor limiar pra cada canal e interpola canais tb
from autoreject import AutoReject

#usar so o limiar aprendido, sem a interpolação

for suj in sujeitos:
    ar = AutoReject()
    ar.fit(dados[suj][:11])
    dados_ar, logs_ar = ar.transform(dados[suj], return_log=True)
    dados[suj][logs_ar.bad_epochs].plot(scalings=dict(eeg=40e-6))
    logs_ar.plot('horizontal')
    
#%%
dados_ar.plot()

#%% REMOCAO DE COMPONENTES COM ICALABEL
#o ICALabel é mais documentado so no caso de dados filtrados entre 1 e 100 e com referencia average

ica = mne.preprocessing.ICA(
    n_components=20,
    method="infomax",
    fit_params=dict(extended=True)
    )

for suj in sujeitos: #usando o epoch que ja rejeitou os artefatos transientes mais fortes
    epoch_ica = copia[suj].copy().filter(1., 100.) #fora isso, o dado ja ta com referencia na media
    ica.fit(epoch_ica)
    ic_labels = label_components(epoch_ica, ica, method='iclabel')
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    print(f"excluindo componentes: {exclude_idx}")
    ica.apply(copia[suj], exclude=exclude_idx)

#%%
print(labels)
#%%
ica.plot_components()
#%%

dados[1].plot()
dados[2].plot()

#dados[1] -> trials: 19, 34, 35, 36, 37, 38, 39, 40, 51, 54, 62, 70, 71, 75, 85, 86, 87
#dados[1] -> eletrodos: CB1

#dados[2] -> trials: 6,7,12,17,19,20,65,66,67,68,69,70,75,80,94,112,113,114,120
#dados[2] -> eletrodos: F6, F8, T7, T8, TP7

#%%

dados[1].drop_bad()
dados[2].drop_bad()

dados[1].drop_channels(epochs.info['bads'])
dados[2].drop_channels(epochs.info['bads'])


#%% vendo se ha algum componente de artefato em freqs menores
dados_ica = {}

dados_ica[1] = dados[1].copy()
dados_ica[2] = dados[2].copy()

ica1 = mne.preprocessing.ICA(n_components = 20, random_state=97)
ica2 = mne.preprocessing.ICA(n_components = 20, random_state=97)

ica1.fit(dados_ica[1])
ica2.fit(dados_ica[2])

ica1.plot_components()
ica2.plot_components()

#%%
ica1.exclude = [0, 1, 17, 19]
ica2.exclude = [0, 8, 12, 19]

ica1.apply(dados_ica[1])
ica2.apply(dados_ica[2])

#%% identificacao de EMG

dados_ica1 = dados_ica[1].copy()
dados_ica1.filter(12., None)

dados_ica2 = dados_ica[2].copy()
dados_ica2.filter(12., None)

ica1 = mne.preprocessing.ICA(n_components = 20, random_state=97)
ica2 = mne.preprocessing.ICA(n_components = 20, random_state=97)

ica1.fit(dados_ica1)
ica2.fit(dados_ica2)

ica1.plot_components()
ica2.plot_components()

#%% remocao de EMG

ica1.exclude = [0, 2, 3, 6, 9, 11, 12, 14, 15, 16, 18, 19]
ica2.exclude = [5, 7, 9, 14, 15, 16, 17, 18,]

ica1.apply(dados_ica[1])
ica2.apply(dados_ica[2])

dados_ica[1].filter(1., 35.)
dados_ica[2].filter(1., 35.)

#%%

dados_ssp = {}
raws_ssp = {}

raws_ssp[1] = dados[1].get_data().copy().as_raw()
raws_ssp[2] = dados[2].get_data().copy().as_raw()

raws_ssp[1].filter(1., 35.)
raws_ssp[2].filter(1., 35.)

dados_ssp[1] = Epochs(raws_ssp[1], events.astype(int), event_id=event_id, picks=picks, preload=True, baseline=(0, 0), tmin=-0.5, tmax=3)
dados_ssp[2] = Epochs(raws_ssp[2], events.astype(int), event_id=event_id, picks=picks, preload=True, baseline=(0, 0), tmin=-0.5, tmax=3)

#%%

dados_ssp[1].plot()
dados_ssp[2].plot()
#%%
dados_csd = {}

for suj in sujeitos:
    dados_csd[suj] = mne.preprocessing.compute_current_source_density(dados[suj].copy())

#%% tfr

pots = {}

for suj in sujeitos:
    pots[suj] = {}

freqs = np.arange(8, 30, 1)  # bandas mu e beta
n_cycles = freqs / 2.  # número de ciclos por frequência

for suj in sujeitos:
    epochs_esq = dados[suj]["left"]
    epochs_dir = dados[suj]["right"]
    
    pot_esq = epochs_esq.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False
    )

    pot_dir = epochs_dir.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=False,
        return_itc=False
    )
    
    pot_esq.apply_baseline(mode="logratio", baseline=(-4, -1.5))
    pot_dir.apply_baseline(mode="logratio", baseline=(-4, -1.5))
    
    pots[suj]['esq'] = pot_esq
    pots[suj]['dir'] = pot_dir

#%%
epochs_esq = dados[1]["left"]
epochs_dir = dados[1]["right"]

pot_esq = epochs_esq.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    average=False,
    return_itc=False
)

pot_dir = epochs_dir.compute_tfr(
    method="morlet",
    freqs=freqs,
    n_cycles=n_cycles,
    average=False,
    return_itc=False
)

pot_esq.apply_baseline(mode="logratio", baseline=(-0.5, -0.1))
pot_dir.apply_baseline(mode="logratio", baseline=(-0.5, -0.1))

pots[1]['esq'] = pot_esq
pots[1]['dir'] = pot_dir


#%% inferencia
import scipy.stats
from mne.stats import permutation_cluster_1samp_test

#aqui nao pode usar o tfr com average, tem que deixar o param average=False na hora de gerar o tfr

#p fazer o teste precisamos de uma matriz de adj dos canais
adj, nomes_ch = mne.channels.find_ch_adjacency(pots[1]['esq'].info, 'eeg')

#trecho veio direto do tutorial
use_idx = [nomes_ch.index(ch_name) for ch_name in pots[1]['esq'].ch_names]
adj = adj[use_idx][:, use_idx]

adjacency = mne.stats.combine_adjacency(
    adj, len(pots[1]['esq'].freqs), len(pots[1]['esq'].times)
)

#%%
tail=0

degrees_of_freedom = len(dados[1]) - 1
t_thresh = scipy.stats.t.ppf(1 - 0.001 / 2, df=degrees_of_freedom)

n_permutations = 5

T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
    pots[1]['esq'].data,
    n_permutations=n_permutations,
    threshold=t_thresh,
    tail=tail,
    adjacency=adjacency,
    out_type="mask",
    verbose=True,
)
#%% gambiarra pra plotar a tfr do laplaciano no prox bloco
mapping = {ch: 'eeg' for ch in ch_names}

for suj in sujeitos:
    for ch in dados_csd[suj].info['ch']:
        ch['unit'] = mne.io.constants.FIFF.FIFF_UNIT_V
        ch['unit_mul'] = 0

dados_csd[1].set_channel_types(mapping)    
dados_csd[2].set_channel_types(mapping)    
#%% tfr do laplaciano

freqs = np.arange(8, 30, 1)  # bandas mu e beta
n_cycles = freqs / 2.  # número de ciclos por frequência

pots_csd = {}

for suj in sujeitos:
    pots_csd[suj] = {}

for suj in sujeitos:
    epochs_esq = dados_csd[suj]["left"]
    epochs_dir = dados_csd[suj]["right"]
    
    pot_esq = epochs_esq.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=False
    )

    pot_dir = epochs_dir.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=False
    )
    
    pots_csd[suj]['esq'] = pot_esq
    pots_csd[suj]['dir'] = pot_dir
    
    pot_esq.plot_topo(baseline=(-0.5, -0.2), mode='logratio', title='espectograma por canal (esq)')
    pot_dir.plot_topo(baseline=(-0.5, -0.2), mode='logratio', title='espectograma por canal (dir)')


#%%

times_topo = np.arange(0., 2.5, 0.2)

for t in times_topo:
    pots_csd[2]['esq'].plot_topomap(
        tmin=t, tmax=(t+.2),
        fmin=12, fmax=28,
        ch_type='eeg',
        baseline=(-0.5, -0.2), mode='logratio',
    )
    
#%%
for t in times_topo:
    pot_esq.plot_topomap(
        tmin=t, tmax=(t+.2),
        fmin=12, fmax=28,
        ch_type='eeg',
        baseline=(-0.5, -0.2), mode='logratio',
    )