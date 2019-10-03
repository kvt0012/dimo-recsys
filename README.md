## Recommender System Microservices

### Install Requirements
```
pip3 install git+https://github.com/maciejkula/spotlight.git@master#egg=spotlight
pip3 install implicit
pip3 install pandas
pip3 install numpy

# In case you use CPU
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Usages:
#### Arguments
- **model_type:** factorization, sequence
- **request_type:**
  - inference (test model's inference ability)
  - update (request api reload for new model)
#### Scripts
- Start API
```
python3 start_api.py -t <model_type> 
```

- Test API
```
python3 test_api.py -t <model_type> -r <request_type>
```

- Tuning Model:
```
python3 tuning.py -t <model_type> 
```
- Reload dataset: load interaction dataset from PostgreSQL Server
```
python3 reload_dataset.py
```
