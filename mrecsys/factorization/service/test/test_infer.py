import socket


def recommend():
    # the return will be in bytes, so decode
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(("localhost", 6000))

    data = {
        "service_name": "FACTORIZATION",
        "service_token": "98f6f754bd9840503459f832d4b243ba36351ec8",
        "data": {
            # "request": "rank_items",
            "request": "recommend",
            "inputs": {
                "user_id": "1000125153834388186",
                "selected_items": ["1023280480942069518",
                                   "306446194545434834",
                                   "3205701174830506022",
                                   "3217347275587941674"],
                "filter_items": ["6481221346259782889"]
            }
        }
    }
    soc.send(str(data).encode("utf8"))
    result_bytes = soc.recv(4096)
    result_string = result_bytes.decode("utf8")
    print("Result from server is {}".format(result_string))


if __name__ == '__main__':
    recommend()