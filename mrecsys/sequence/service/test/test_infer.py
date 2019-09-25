import socket
import time


if __name__ == '__main__':
    # the return will be in bytes, so decode
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(("localhost", 5000))

    data = {
        "service_name": "SEQUENCE",
        "service_token": "ee977806d7286510da8b9a7492ba58e2484c0ecc",
        "data": {
            "request": "rank_items",
            "inputs": {
                "sequence": ["100459171840484639", "100670421557790535"],
                "selected_items": ["1023280480942069518",
                                   "306446194545434834",
                                   "3205701174830506022",
                                   "3217347275587941674"],
                "filter_items": ["3217347275587941674"]
            }
        }
    }
    soc.send(str(data).encode("utf8"))
    result_bytes = soc.recv(4096)
    result_string = result_bytes.decode("utf8")
    print("Result from server is {}".format(result_string))
    time.sleep(2)
