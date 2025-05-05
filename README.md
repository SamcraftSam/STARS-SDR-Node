
# STARS-Node

**Satellite Tracking And Receiving System** – an open-source Python utility that transforms a Raspberry Pi (or any Linux box) with an SDR (like HackRF or RTL-SDR) into an autonomous satellite data receiver. This tool is tailored for receiving NOAA APT signals, decoding them into images, and optionally colorizing them with LUT palettes. 

## 🚀 Introduction

STARS-Node was built with the goal of making NOAA satellite image reception as dead simple and modular as possible. The main motivation behind this project was to explore signal processing using object-oriented principles in Python while building a practical and useful tool.

## 🧠 Problem & Requirements

**Problem Solved:**  
Automating the reception, demodulation, decoding, and visualization of APT images broadcasted by NOAA weather satellites.

**Functional Requirements:**
- Receive baseband I/Q samples from SDR or WAV file.
- Demodulate and decode NOAA APT signals.
- Save grayscale and colorized satellite images.
- Provide testable modular DSP components.

**Non-Functional Requirements:**
- Platform agnostic (Linux focus).
- Modular, testable code.
- Lightweight enough to run on Raspberry Pi.

## ⚙️ Design and Implementation

This project uses classic OOP principles:
- **Abstraction** via `GenericModule` and `GenericSink` base classes.
- **Encapsulation** of DSP pipeline components.
- **Inheritance** to extend generic DSP processing into specific filters, demodulators, and decoders.
- **Polymorphism** used throughout processing pipelines and sinks.

### Main Components

- `dsptools.py`: Modular building blocks (filters, decoders, sinks).
- `apt_decoder.py`: Legacy NOAA APT decoding logic, adapted for batch and stream decoding.
- `apt_colorize.py`: Applies a 2D LUT to grayscale APT images to enhance visual quality.
- `main.py`: Pipeline runner for offline WAV decoding and image generation.

### Class Structure

The system is organized into modules following SRP, with core pipelines reusable and testable. Decoder classes implement a clear `__call__` and `decode()` interface.

## 🛠 Development Process

**Languages & Tools:**
- Python 3
- NumPy, SciPy, PIL, Matplotlib
- pytest for testing
- sounddevice for audio sink support
- SDR++ for raw baseband recording

**Environment:**
- Developed on Linux (Debian-based)
- SDR tested with HackRF and RTL-SDR dongles

**Steps Taken:**
1. Decoded NOAA APT from WAV manually to validate baseline.
2. Modularized DSP stages.
3. Built reusable pipeline classes.
4. Integrated image colorization with LUTs.
5. Wrote unit tests for all core components.

## 📸 Features & Demo

- NOAA APT decoding (grayscale and color)
- Modular DSP chain (WAV→IQ→FM→APT)
- Headless batch processing
- Unit-tested DSP blocks
- Image colorization with 2D LUT

![NOAA APT Sample](./noaa-apt-daylight.png)

## ✅ Testing

All critical DSP and transformation components are covered via `pytest`-based unit tests:

- Colorizer behavior with 2D arrays and files
- WAV/byte to complex conversions
- Filter responses and demodulators
- Coordinate conversion logic

Issues like invalid input types, odd image dimensions, and filter edge cases are also covered.

## 🧾 Conclusion & Future Work

STARS-Node demonstrates how Python and SDR can be combined in a clean, extensible architecture for satellite image decoding.

**Future Enhancements:**
- Add real-time streaming support for live decoding.
- Implement GUI for visualizing pass data.
- Support other satellite formats (e.g., Meteor-M2, LRPT).
- Improve Doppler correction in real-time mode.

---

*Built for nerds. Runs on Pi. Sniffs the sky.*
