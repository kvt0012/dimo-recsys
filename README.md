# DevC Innovation Challenge - HKTTP Team

## Recommender System Microservices for Dimo App
Consist of two types of Recommender Model
  - **Factorization**: For Personalization by using Alternating Least Square powered by Implicit library
  - **Sequence**: For handle the problem "what's the next item to recommend for user based on sequence of his interacted items" (Session-based recommender system) powered by Spotlight library (using Pytorch)

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
- Training Model:
```
python3 training.py -t <model_type> 
```

- Tuning Model:
```
python3 tuning.py -t <model_type> 
```
- Reload dataset: load interaction dataset from PostgreSQL Server
```
python3 reload_dataset.py
```
