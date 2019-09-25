from mrecsys.utils.deploy import request_update

if __name__ == '__main__':
    print(request_update(service_ip="localhost",
                         service_port=5000,
                         service_name="SEQUENCE",
                         service_token="ee977806d7286510da8b9a7492ba58e2484c0ecc"))
