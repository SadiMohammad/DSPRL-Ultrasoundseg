# DSPRL-Ultrasoundseg
**Train**
1. cd to `./codes`

2. run the following command for training </br>
&nbsp;  `python train.py --config_filename <config filename with path> --config_scheme <section from config file>`</br>
&nbsp; &nbsp;  e.g.&nbsp; ```python train.py --config_filename config.ini --config_scheme DEFAULT```

**Inference**
1. cd to `./codes`

2. run the following command for inference</br>
&nbsp;  `python infer.py --config_filename <config filename with path> --config_scheme <section from config file>`</br>
&nbsp; &nbsp;  e.g.&nbsp; ```python infer.py --config_filename config.ini --config_scheme TEST```

**Inference with MorphSnake**
1. cd to `./codes`

2. run the following command for inference</br>
&nbsp;  `python inferWithMS.py --config_filename <config filename with path> --config_scheme <section from config file>`</br>
&nbsp; &nbsp;  e.g.&nbsp; ```python inferWithMS.py --config_filename config.ini --config_scheme TEST```
