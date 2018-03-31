import json
import socket

import neuralData

port = 9090
host = "localhost"

# print(json.dumps(neuralData.neural_data, ensure_ascii=False))

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))

sock.listen(1)
conn, addr = sock.accept()

while True:
    data = conn.recv(2048)
    message = data.decode('ascii')
    print("message")
    print(message)
    print("received data")
    ans = json.loads(message)
    print(ans)

    if not data:
        break

    output = json.dumps(neuralData.neural_data,
               ensure_ascii=True)

    print("send data")
    print(output)
    outdata = output.encode()
    print(outdata)
    conn.send(outdata)

conn.close()
