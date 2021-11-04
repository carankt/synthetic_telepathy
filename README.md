# synthetic_telepathy
Synthetic Telepathy: Inner Speech Recognition using EEG

## Dataset Used 
Inner Speech - Nicolas Nieto and Victoria Peterson and Hugo Rufiner and Juan Kamienkowski and Ruben Spies (2021). Inner Speech. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003626.v2.0.0 [Download here](https://openneuro.org/datasets/ds003626/versions/2.0.0) 

## Dataloader Functionality

```python
from dataloader import get_loader
root = '/path/to/inner_speech/derivatives_folder/'
creater = get_loader(root)
xn, yn = creater.load_multiple_subjects([1, 2, 3, 4, 5, 6, 7, 8]) # loads 8 subjects data and stacks them in a 3d array
```

* xn - shape - (number_of_samples, channels, time_stamps)
* yn - shape - (number_of_samples, field_info) - for classes choose yn[:, 1]


