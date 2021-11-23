# synthetic_telepathy
Synthetic Telepathy: Inner Speech Recognition using EEG

## Dataset Used 
Inner Speech - Nicolas Nieto and Victoria Peterson and Hugo Rufiner and Juan Kamienkowski and Ruben Spies (2021). Inner Speech. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003626.v2.0.0 [Download here](https://openneuro.org/datasets/ds003626/versions/2.0.0) 

## Installing all the requirements 

```python
pip install -r requirements.txt
```


## Dataloader Functionality

```python
from dataloader import get_loader
root = '/Volumes/Datasets/inner_speech/derivatives/'
# root =  'dataset/derivatives/' # -sil
creater = get_loader(root, channel_list= ["A4", "A5", "A19", "A20", "A32"], n_sess= 3)
xn, yn = creater.load_multiple_subjects([1, 2, 4, 5, 6, 7])
```

* xn - shape - (number_of_samples, channels, time_stamps)
* yn - shape - (number_of_samples, field_info) - for classes choose yn[:, 1]


