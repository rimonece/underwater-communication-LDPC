**Underwater Communication System Simulation**
This repository provides a simulation framework for evaluating the performance of underwater acoustic communication systems. It includes support for multiple modulation schemes, LDPC error correction coding, and multi-hop transmission scenarios over an underwater acoustic channel.

**Network Scenario**
This simulation models an underwater communication system with:

Multiple Nodes: Nodes placed at varying distances and depths.
Underwater Acoustic Channel: Introduces noise and attenuation, with adjustable Signal-to-Noise Ratio (SNR).
Modulation Schemes: Supports BPSK, QPSK, FSK, and 16-QAM.
Error Correction: Uses Low-Density Parity Check (LDPC) coding for reliable data transmission.
Performance Metrics:
Bit Error Rate (BER): Fraction of bits decoded incorrectly.
Frame Error Rate (FER): Fraction of frames with errors.
Success Rate: Percentage of error-free transmissions.

**Features**
Simulates multi-hop underwater communication with random node placements.
Evaluates BER and FER across different modulation schemes and SNR values.
Visualizes performance metrics through plots for easy analysis.
Includes a flexible and modular design for future extensions.

**Generated Outputs**
The simulation generates the following plots:

BER vs. SNR:
Compares bit error rate across modulation schemes for different SNR levels.
FER vs. SNR:
Illustrates frame error rate trends for all modulation schemes.

**Getting Started**
Prerequisites
Python 3.x
Required Python libraries:
numpy
matplotlib
scipy
tqdm

Install the dependencies using:

1. Navigate to the project directory:
cd underwater-communication-simulation

2. Run the simulation:
python underwater_simulation.py

3. View the generated plots for BER and FER analysis.


**Code Overview**

**Key Components**

DigitalDemodulator:

Supports demodulation for BPSK, QPSK, FSK, and 16-QAM.
Includes signal amplification, gain adjustment, and equalization.
UnderwaterChannel:

Simulates an underwater acoustic channel with noise and attenuation.
Allows transmission at adjustable SNR levels.
DigitalModulator:

Encodes binary data into modulated waveforms.
LDPC:

Implements Low-Density Parity Check encoding and decoding.
Main Simulation (test_combined_ldpc):

Simulates transmission and reception for multiple SNR levels and modulation schemes.
Outputs performance metrics (BER, FER) for analysis.

**Output Examples**
1. BER vs. SNR
2. FER vs. SNR

Future Extensions
Incorporate advanced channel models with multi-path effects.
Add support for adaptive modulation and coding.
Explore higher-order modulation schemes like 64-QAM or OFDM.
Contributing
Contributions are welcome! If you'd like to contribute:


License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements:
This simulation framework was developed to support research and educational purposes of contributing to error correction in underwater communication systems. Your feedback and contributions are highly appreciated!
