from mrecsys.utils.deploy import request_update

if __name__ == '__main__':
    print(request_update(service_ip="localhost",
                         service_port=6000,
                         service_name="FACTORIZATION",
                         service_token="98f6f754bd9840503459f832d4b243ba36351ec8"))
