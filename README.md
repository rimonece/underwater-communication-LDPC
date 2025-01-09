# **Underwater Communication System Simulation**

This repository contains a simulation framework for evaluating the performance of underwater acoustic communication systems. It supports various modulation schemes, error correction techniques, and performance metrics to analyze the efficiency and reliability of underwater communications.

---

## **Network Scenario**

The simulation models a multi-hop underwater communication system with:
1. **Nodes**:
   - Randomly placed nodes with varying distances and depths.
2. **Underwater Acoustic Channel**:
   - Simulates noise and attenuation, with adjustable Signal-to-Noise Ratio (SNR).
3. **Modulation Schemes**:
   - **BPSK**, **QPSK**, **FSK**, and **16-QAM**.
4. **Error Correction**:
   - Low-Density Parity Check (LDPC) coding to improve reliability.
5. **Performance Metrics**:
   - **Bit Error Rate (BER):** Fraction of bits decoded incorrectly.
   - **Frame Error Rate (FER):** Fraction of frames with errors.
   - **Success Rate:** Percentage of error-free transmissions.

---

## **Features**

- Multi-hop underwater acoustic communication simulation.
- Evaluation of BER and FER across multiple modulation schemes and SNR values.
- Visualizations for performance analysis.

---

## **Generated Outputs**
Run the main_uw_ldpc.py

## **Generated Outputs**

The simulation generates the following plots:
1. **BER vs. SNR**:
   - Compares bit error rates for all modulation schemes.
2. **FER vs. SNR**:
   - Illustrates frame error rates across modulation schemes.

---

## **Getting Started**

### **Prerequisites**
- Python 3.x
- Required Python libraries:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `tqdm`

### **Code Overview**

#### **Key Components**

- **`DigitalDemodulator`**:
  - Supports demodulation for **BPSK**, **QPSK**, **FSK**, and **16-QAM**.
  - Includes signal amplification, gain adjustment, and equalization.

- **`UnderwaterChannel`**:
  - Simulates an underwater acoustic channel with noise and attenuation.
  - Allows transmission at adjustable SNR levels.

- **`DigitalModulator`**:
  - Encodes binary data into modulated waveforms.

- **`LDPC`**:
  - Implements Low-Density Parity Check encoding and decoding.

- **Main Simulation (`test_combined_ldpc`)**:
  - Simulates transmission and reception for multiple SNR levels and modulation schemes.
  - Outputs performance metrics (**BER**, **FER**) for analysis.

---

### **Output Examples**

1. **BER vs. SNR**
2. **FER vs. SNR**

---

### **Future Extensions**

1. Incorporate advanced channel models with multi-path effects.
2. Add support for adaptive modulation and coding.
3. Explore higher-order modulation schemes like **64-QAM** or **OFDM**.

---

### **Contributing**

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name

### **Acknowledgements**

This simulation framework was developed to advance research and education for error correction using LPDPC coding in underwater communication systems. It aims to support the study of error correction techniques, modulation schemes, and reliable communication in challenging underwater environments.
