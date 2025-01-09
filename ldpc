import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.signal import correlate

# Configure logging
logging.basicConfig(filename='combined_results.log', filemode='w', level=logging.DEBUG)


### Digital Demodulator Implementation
class DigitalDemodulator:
    def __init__(self, modulation_type='BPSK', f1=2500, f0=4000, sample_rate=10000, bit_time=8):
        self.modulation_type = modulation_type
        self.f1 = f1
        self.f0 = f0
        self.sample_rate = sample_rate
        self.bit_time = bit_time

    def demodulate(self, received_signal, gain=1.0, equalize=False, dynamic_gain=False):
        if dynamic_gain:
            received_signal = self.adjust_gain(received_signal)

        amplified_signal = self.amplify_signal(received_signal, gain)

        if equalize:
            amplified_signal = self.adaptive_equalize_signal(amplified_signal)

        if self.modulation_type == 'BPSK':
            return self.bpsk_demodulate(amplified_signal)
        elif self.modulation_type == 'QPSK':
            return self.qpsk_demodulate(amplified_signal)
        elif self.modulation_type == 'FSK':
            return self.fsk_demodulate(amplified_signal)
        elif self.modulation_type == '16-QAM':
            return self.qam_demodulate(amplified_signal, 16)
        else:
            raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

    def amplify_signal(self, signal, gain):
        return signal * gain

    def adjust_gain(self, signal):
        mean_signal = np.mean(signal)
        if mean_signal < 0.1:
            return signal * 10
        return signal

    def adaptive_equalize_signal(self, signal):
        num_taps = 5
        equalizer = np.ones(num_taps) / num_taps
        return np.convolve(signal, equalizer, mode='same')

    def bpsk_demodulate(self, received_signal):
        return (np.array(received_signal) > 0).astype(int)

    def qpsk_demodulate(self, received_signal):
        demodulated_bits = []
        for symbol in received_signal:
            real_part = np.real(symbol)
            imag_part = np.imag(symbol)
            if real_part > 0 and imag_part > 0:
                demodulated_bits.extend([0, 0])
            elif real_part < 0 and imag_part > 0:
                demodulated_bits.extend([0, 1])
            elif real_part > 0 and imag_part < 0:
                demodulated_bits.extend([1, 0])
            elif real_part < 0 and imag_part < 0:
                demodulated_bits.extend([1, 1])
        return np.array(demodulated_bits)

    def fsk_demodulate(self, received_signal):
        samples_per_bit = int(self.bit_time * self.sample_rate)
        num_bits = len(received_signal) // samples_per_bit
        demodulated_bits = np.zeros(num_bits, dtype=int)

        ref_signal_0 = np.sin(2 * np.pi * self.f0 * np.arange(0, self.bit_time, 1 / self.sample_rate))
        ref_signal_1 = np.sin(2 * np.pi * self.f1 * np.arange(0, self.bit_time, 1 / self.sample_rate))

        for i in range(num_bits):
            segment = received_signal[i * samples_per_bit:(i + 1) * samples_per_bit]
            corr_0 = np.max(correlate(segment, ref_signal_0))
            corr_1 = np.max(correlate(segment, ref_signal_1))
            demodulated_bits[i] = 0 if corr_0 > corr_1 else 1

        return demodulated_bits

    def qam_demodulate(self, received_signal, M):
        k = int(np.log2(M))
        num_symbols = len(received_signal)
        demodulated_bits = np.zeros(num_symbols * k, dtype=int)
        for i, received_symbol in enumerate(received_signal):
            I_estimate = np.round(received_symbol.real)
            Q_estimate = np.round(received_symbol.imag)
            I_estimate = np.clip(I_estimate, -(k - 1), k - 1)
            Q_estimate = np.clip(Q_estimate, -(k - 1), k - 1)

            symbol_index = ((I_estimate + 3) // 2) + 4 * ((Q_estimate + 3) // 2)
            bits = np.array(list(np.binary_repr(int(symbol_index), width=k)), dtype=int)
            demodulated_bits[i * k:(i + 1) * k] = bits

        return demodulated_bits


### Mock UnderwaterChannel Implementation
class UnderwaterChannel:
    def __init__(self, frequency):
        self.frequency = frequency

    def multi_hop_transmit(self, modulated_signal, nodes_positions, snr_dB):
        noise = np.random.normal(0, 1, len(modulated_signal)) * 10 ** (-snr_dB / 20)
        received_signal = modulated_signal + noise
        return received_signal, None, None, None


### Mock DigitalModulator Implementation
class DigitalModulator:
    def __init__(self, modulation_type):
        self.modulation_type = modulation_type

    def modulate(self, codeword):
        return np.array([1 if bit == 1 else -1 for bit in codeword])  # Simple BPSK logic


### Mock LDPC Implementation
class LDPC:
    def __init__(self, H):
        self.H = H
        self.n = H.shape[1]
        self.k = self.n - H.shape[0]

    def encode(self, message):
        return np.concatenate([message, np.zeros(self.n - len(message), dtype=int)])

    def decode(self, received_signal):
        return received_signal[:self.k]


### Main Code
def calculate_distance(node):
    x, y = node
    return np.sqrt(x ** 2 + y ** 2)

import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.signal import correlate

# Configure logging
logging.basicConfig(filename='combined_results.log', filemode='w', level=logging.DEBUG)


### Digital Demodulator Implementation
class DigitalDemodulator:
    def __init__(self, modulation_type='BPSK', f1=2500, f0=4000, sample_rate=10000, bit_time=8):
        self.modulation_type = modulation_type
        self.f1 = f1
        self.f0 = f0
        self.sample_rate = sample_rate
        self.bit_time = bit_time

    def demodulate(self, received_signal, gain=1.0, equalize=False, dynamic_gain=False):
        if dynamic_gain:
            received_signal = self.adjust_gain(received_signal)

        amplified_signal = self.amplify_signal(received_signal, gain)

        if equalize:
            amplified_signal = self.adaptive_equalize_signal(amplified_signal)

        if self.modulation_type == 'BPSK':
            return self.bpsk_demodulate(amplified_signal)
        elif self.modulation_type == 'QPSK':
            return self.qpsk_demodulate(amplified_signal)
        elif self.modulation_type == 'FSK':
            return self.fsk_demodulate(amplified_signal)
        elif self.modulation_type == '16-QAM':
            return self.qam_demodulate(amplified_signal, 16)
        else:
            raise ValueError(f"Unsupported modulation type: {self.modulation_type}")

    def amplify_signal(self, signal, gain):
        return signal * gain

    def adjust_gain(self, signal):
        mean_signal = np.mean(signal)
        if mean_signal < 0.1:
            return signal * 10
        return signal

    def adaptive_equalize_signal(self, signal):
        num_taps = 5
        equalizer = np.ones(num_taps) / num_taps
        return np.convolve(signal, equalizer, mode='same')

    def bpsk_demodulate(self, received_signal):
        return (np.array(received_signal) > 0).astype(int)

    def qpsk_demodulate(self, received_signal):
        demodulated_bits = []
        for symbol in received_signal:
            real_part = np.real(symbol)
            imag_part = np.imag(symbol)
            if real_part > 0 and imag_part > 0:
                demodulated_bits.extend([0, 0])
            elif real_part < 0 and imag_part > 0:
                demodulated_bits.extend([0, 1])
            elif real_part > 0 and imag_part < 0:
                demodulated_bits.extend([1, 0])
            elif real_part < 0 and imag_part < 0:
                demodulated_bits.extend([1, 1])
        return np.array(demodulated_bits)

    def fsk_demodulate(self, received_signal):
        samples_per_bit = int(self.bit_time * self.sample_rate)
        num_bits = len(received_signal) // samples_per_bit
        demodulated_bits = np.zeros(num_bits, dtype=int)

        ref_signal_0 = np.sin(2 * np.pi * self.f0 * np.arange(0, self.bit_time, 1 / self.sample_rate))
        ref_signal_1 = np.sin(2 * np.pi * self.f1 * np.arange(0, self.bit_time, 1 / self.sample_rate))

        for i in range(num_bits):
            segment = received_signal[i * samples_per_bit:(i + 1) * samples_per_bit]
            corr_0 = np.max(correlate(segment, ref_signal_0))
            corr_1 = np.max(correlate(segment, ref_signal_1))
            demodulated_bits[i] = 0 if corr_0 > corr_1 else 1

        return demodulated_bits

    def qam_demodulate(self, received_signal, M):
        k = int(np.log2(M))
        num_symbols = len(received_signal)
        demodulated_bits = np.zeros(num_symbols * k, dtype=int)
        for i, received_symbol in enumerate(received_signal):
            I_estimate = np.round(received_symbol.real)
            Q_estimate = np.round(received_symbol.imag)
            I_estimate = np.clip(I_estimate, -(k - 1), k - 1)
            Q_estimate = np.clip(Q_estimate, -(k - 1), k - 1)

            symbol_index = ((I_estimate + 3) // 2) + 4 * ((Q_estimate + 3) // 2)
            bits = np.array(list(np.binary_repr(int(symbol_index), width=k)), dtype=int)
            demodulated_bits[i * k:(i + 1) * k] = bits

        return demodulated_bits


### Mock UnderwaterChannel Implementation
class UnderwaterChannel:
    def __init__(self, frequency):
        self.frequency = frequency

    def multi_hop_transmit(self, modulated_signal, nodes_positions, snr_dB):
        noise = np.random.normal(0, 1, len(modulated_signal)) * 10 ** (-snr_dB / 20)
        received_signal = modulated_signal + noise
        return received_signal, None, None, None


### Mock DigitalModulator Implementation
class DigitalModulator:
    def __init__(self, modulation_type):
        self.modulation_type = modulation_type

    def modulate(self, codeword):
        return np.array([1 if bit == 1 else -1 for bit in codeword])  # Simple BPSK logic


### Mock LDPC Implementation
class LDPC:
    def __init__(self, H):
        self.H = H
        self.n = H.shape[1]
        self.k = self.n - H.shape[0]

    def encode(self, message):
        return np.concatenate([message, np.zeros(self.n - len(message), dtype=int)])

    def decode(self, received_signal):
        return received_signal[:self.k]


### Main Code with Figure Generation
def test_combined_ldpc(num_nodes):
    ldpc_configs = [np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 0, 0, 1]])]
    snr_values = np.arange(-30, 40, 15)
    modulation_types = ['BPSK', 'QPSK', 'FSK', '16-QAM']
    results = {mod: {"snr": [], "ber": [], "fer": []} for mod in modulation_types}

    num_tests = 10
    underwater_channel = UnderwaterChannel(frequency=30000)
    max_distance, max_depth = 25, 50
    sorted_distances = sorted(np.random.randint(0, max_distance, num_nodes))
    sorted_depths = sorted(np.random.randint(0, max_depth, num_nodes))
    nodes_positions = [(sorted_distances[i], sorted_depths[i]) for i in range(num_nodes)]

    for config_index, H in enumerate(ldpc_configs):
        ldpc = LDPC(H)

        for modulation_type in modulation_types:
            for snr_dB in snr_values:
                bit_errors = 0
                total_bits = 0
                frame_errors = 0

                for _ in tqdm(range(num_tests), desc=f'{modulation_type} at {snr_dB} dB'):
                    message = np.random.randint(2, size=ldpc.k)
                    codeword = ldpc.encode(message)
                    modulator = DigitalModulator(modulation_type)
                    modulated_signal = modulator.modulate(codeword)
                    received_signal, _, _, _ = underwater_channel.multi_hop_transmit(
                        modulated_signal, nodes_positions, snr_dB)
                    demodulator = DigitalDemodulator(modulation_type)
                    decoded_ldpc = demodulator.demodulate(received_signal)

                    if len(decoded_ldpc) < ldpc.n:
                        decoded_ldpc = np.pad(decoded_ldpc, (0, ldpc.n - len(decoded_ldpc)), 'constant')
                    elif len(decoded_ldpc) > ldpc.n:
                        decoded_ldpc = decoded_ldpc[:ldpc.n]

                    decoded_message = ldpc.decode(decoded_ldpc)

                    if len(decoded_message) == len(message):
                        errors = np.sum(decoded_message != message)
                        bit_errors += errors
                        total_bits += len(message)
                        if errors > 0:
                            frame_errors += 1

                ber = bit_errors / total_bits
                fer = frame_errors / num_tests

                results[modulation_type]["snr"].append(snr_dB)
                results[modulation_type]["ber"].append(ber)
                results[modulation_type]["fer"].append(fer)

    # Plot BER and FER
    for mod in modulation_types:
        plt.figure(figsize=(10, 6))
        plt.plot(results[mod]["snr"], results[mod]["ber"], marker='o', label="BER")
        plt.plot(results[mod]["snr"], results[mod]["fer"], marker='x', label="FER")
        plt.title(f"Performance for {mod}")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Error Rate")
        plt.yscale("log")
        plt.grid(True)
        plt.legend()
        plt.show()

            # Plot combined BER
    plt.figure(figsize=(10, 6))
    for mod in modulation_types:
        plt.plot(results[mod]["snr"], results[mod]["ber"], marker='o', label=f"{mod} - BER")
    plt.title("BER vs SNR for All Modulation Schemes")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot combined FER
    plt.figure(figsize=(10, 6))
    for mod in modulation_types:
        plt.plot(results[mod]["snr"], results[mod]["fer"], marker='x', label=f"{mod} - FER")
    plt.title("FER vs SNR for All Modulation Schemes")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Frame Error Rate (FER)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_combined_ldpc(5)
