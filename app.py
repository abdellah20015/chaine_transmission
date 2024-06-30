from flask import Flask, render_template, request, jsonify, Response
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Utilisation du backend 'Agg'

app = Flask(__name__)

# Fonctions de codage en ligne
def RZ(vector):
    for i in range(len(vector)):
        plt.plot([50 * i, 50 * (i + 1)], [vector[i], vector[i]], color='royalblue')
        if i != 0 and vector[i - 1] != vector[i]:
            plt.plot([50 * i, 50 * i], [0, 1], color='royalblue')
    plt.yticks([0, 1])

def NRZ(vector):
    y = [1 if yn == 1 else -1 for yn in vector]
    for i in range(len(y)):
        plt.plot([50 * i, 50 * (i + 1)], [y[i], y[i]], color='royalblue')
        if i != 0 and y[i - 1] != y[i]:
            plt.plot([50 * i, 50 * i], [-1, 1], color='royalblue')
    plt.yticks([-1, 0, 1])

def Manchester(vector):
    for i in range(len(vector)):
        if i != 0 and vector[i] == vector[i - 1]:
            plt.plot([50 * i, 50 * i], [1, -1], color='royalblue')
        if vector[i] == 1:
            plt.plot([50 * i, 50 * (i + 0.5)], [1, 1], color='royalblue')
            plt.plot([50 * (i + 0.5), 50 * (i + 0.5)], [1, -1], color='royalblue')
            plt.plot([50 * (i + 0.5), 50 * (i + 1)], [-1, -1], color='royalblue')
        else:
            plt.plot([50 * i, 50 * (i + 0.5)], [-1, -1], color='royalblue')
            plt.plot([50 * (i + 0.5), 50 * (i + 0.5)], [1, -1], color='royalblue')
            plt.plot([50 * (i + 0.5), 50 * (i + 1)], [1, 1], color='royalblue')
        plt.yticks([-1, 0, 1])

def Miller(vector):
    pos = 0
    for i in range(len(vector)):
        if vector[i] == 0:
            plt.plot([50 * i, 50 * (i + 1)], [0, 0], color='royalblue')
        else:
            if pos == 0:
                plt.plot([50 * i, 50 * i], [0, 1], color='royalblue')
                plt.plot([50 * i, 50 * (i + 1)], [1, 1], color='royalblue')
                plt.plot([50 * (i + 1), 50 * (i + 1)], [0, 1], color='royalblue')
            else:
                plt.plot([50 * i, 50 * i], [0, -1], color='royalblue')
                plt.plot([50 * i, 50 * (i + 1)], [-1, -1], color='royalblue')
                plt.plot([50 * (i + 1), 50 * (i + 1)], [0, -1], color='royalblue')
            pos = 0 if pos == 1 else 1
        plt.yticks([-1, 0, 1])

def encode_image():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/encode', methods=['POST'])
def encode():
    data = request.json
    code = data['code']
    message = data['message']
    vector = [int(i) for i in message]
    plt.figure()
    if code == "RZ":
        RZ(vector)
    elif code == "NRZ":
        NRZ(vector)
    elif code == "Manchester":
        Manchester(vector)
    elif code == "Miller":
        Miller(vector)
    plt.grid(True)
    plt.title(f"Code en ligne: {code}")
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/filtreEmission', methods=['POST'])
def filtreEmission():
    data = request.json
    message = data['message']
    vector = [int(i) for i in message]
    beta = 0.35
    Ts = 16
    t = np.arange(-50, 51)
    h = 1 / Ts * np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    x = np.array([])
    for bit in vector:
        pulse = np.zeros(Ts)
        pulse[0] = bit
        x = np.concatenate((x, pulse))
    x_shaped = np.convolve(x, h)
    plt.figure()
    plt.plot(x_shaped, color='royalblue')
    plt.grid(True)
    plt.title("Filtre d'émission")
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

def ASK_ook(data):
    A = 15
    tb = 0.6
    t = 0
    s = []
    fc = 0.2
    while t <= len(data):
        s.append(data[int(t)] * A * np.sin(2 * np.pi * fc * t))
        t = t + tb
    plt.plot(s, color='royalblue')
    return s

def psk(data):
    A = 15
    tb = 0.01
    t = 0
    s = []
    fc = 0.2
    phi = 90
    while t <= len(data):
        s.append(data[int(t)] * np.sin(2 * np.pi * fc * t + np.deg2rad(phi)))
        t = t + tb
    plt.plot(s, color='royalblue')
    return s

def fsk(data):
    f1 = 0.1
    f2 = 0.16
    tb = 0.01
    t = 0
    s = []
    Tb = 0.01
    df = f2 - f1
    f0 = (f1 + f2) / 2
    while t <= len(data):
        if data[int(t)] > 0.01:
            s.append(data[int(t)] * np.sin(2 * np.pi * f0 * t + (np.pi * df * t)))
        else:
            s.append(data[int(t)] * np.sin(2 * np.pi * f0 * t - (np.pi * df * t)))
        t = t + tb
    plt.plot(s, color='royalblue')
    return s

@app.route('/modulation', methods=['POST'])
def modulation():
    data = request.json
    mod = data['mod']
    message = data['message']
    vector = [int(i) for i in message]
    beta = 0.35
    Ts = 16
    t = np.arange(-50, 51)
    h = 1 / Ts * np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    x = np.array([])
    for bit in vector:
        pulse = np.zeros(Ts)
        pulse[0] = bit
        x = np.concatenate((x, pulse))
    x_shaped = np.convolve(x, h)
    plt.figure()
    if mod == "ASK":
        s = ASK_ook(x_shaped)
    elif mod == "PSK":
        s = psk(x_shaped)
    elif mod == "FSK":
        s = fsk(x_shaped)
    plt.title(f"Modulation {mod}")
    plt.grid()
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/canalPropagation', methods=['POST'])
def canalPropagation():
    data = request.json
    mod = data['mod']
    message = data['message']
    vector = [int(i) for i in message]
    beta = 0.35
    Ts = 16
    t = np.arange(-50, 51)
    h = 1 / Ts * np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    x = np.array([])
    for bit in vector:
        pulse = np.zeros(Ts)
        pulse[0] = bit
        x = np.concatenate((x, pulse))
    x_shaped = np.convolve(x, h)
    if mod == "ASK":
        s = ASK_ook(x_shaped)
    elif mod == "PSK":
        s = psk(x_shaped)
    elif mod == "FSK":
        s = fsk(x_shaped)
    sigma = 0.1
    mean = 0
    noise = np.random.normal(mean, sigma, len(s))
    signalN = s + noise
    plt.figure()
    plt.plot(signalN, color='royalblue')
    plt.title("Sortie du canal de propagation")
    plt.grid()
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/demodulation', methods=['POST'])
def demodulation():
    data = request.json
    message = data['message']
    vector = [int(i) for i in message]
    beta = 0.35
    Ts = 16
    t = np.arange(-50, 51)
    h = 1 / Ts * np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    x = np.array([])
    for bit in vector:
        pulse = np.zeros(Ts)
        pulse[0] = bit
        x = np.concatenate((x, pulse))
    x_shaped = np.convolve(x, h)
    sigma = 0.0025
    mean = 0
    noise = np.random.normal(mean, sigma, len(x_shaped))
    demo = x_shaped + noise
    plt.figure()
    plt.plot(demo, color='royalblue')
    plt.title("Demodulation")
    plt.grid()
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/filtreReception', methods=['POST'])
def filtreReception():
    data = request.json
    message = data['message']
    vector = [int(i) for i in message]
    beta = 0.35
    Ts = 16
    t = np.arange(-50, 51)
    h = 1 / Ts * np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    x = np.array([])
    for bit in vector:
        pulse = np.zeros(Ts)
        pulse[0] = bit
        x = np.concatenate((x, pulse))
    x_shaped = np.convolve(x, h)
    plt.figure()
    plt.plot(x_shaped, color='royalblue')
    plt.grid(True)
    plt.title("Filtre de reception")
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/decision', methods=['POST'])
def decision():
    data = request.json
    code = data['code']
    message = data['message']
    vector = [int(i) for i in message]
    plt.figure()
    if code == "RZ":
        RZ(vector)
    elif code == "NRZ":
        NRZ(vector)
    elif code == "Manchester":
        Manchester(vector)
    elif code == "Miller":
        Miller(vector)
    plt.grid(True)
    plt.title(f"Signal après la decision: {code}")
    img_str = encode_image()
    plt.close()
    return jsonify({'image': img_str})

@app.route('/clear', methods=['POST'])
def clear():
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(debug=True)
