# DSPRL-Ultrasoundseg
**Train**
1. cd to codes

2. run the following command for training 

`python train.py --config_filename <config filename with path> --config_scheme <section from config file>`
example : ```python train.py --config_filename config.ini --config_scheme DEFAULT```

**Inference**
1. cd to codes

2. run the following command for inference
`python inferWithMS.py --config_filename <config filename with path> --config_scheme <section from config file>`
example : ```python inferWithMS.py --config_filename config.ini --config_scheme TEST```