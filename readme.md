# AI-Powered Weapon Detection System

## Overview
An advanced Artificial Intelligence system designed to detect weapons in real-time using computer vision and deep learning technologies. This project leverages state-of-the-art AI models for accurate weapon identification and threat assessment.

## Features
- **Real-time AI Detection**: Continuous monitoring using advanced neural networks
- **Multi-weapon Classification**: Detects multiple types of weapons (knives, pistols, rifles)
- **Audio Alert System**: Integrated alert mechanism for identified threats
- **Thread-safe Processing**: Optimized multi-threaded architecture for efficient processing

## Tech Stack
- **AI Framework**: OpenCV DNN
- **Programming Language**: Python 3.x
- **Deep Learning Model**: YOLO (You Only Look Once)
- **Audio Processing**: Pygame
- **Concurrency**: Python Threading

## Installation
```bash
# Clone the repository
git clone https://github.com/aayush-ojha/ai-weapon-detection.git

# Navigate to project directory
cd ai-weapon-detection

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python main.p
```

## Model Architecture
The system utilizes a custom-trained AI model based on the YOLO architecture for weapon detection:
- Input Layer: 416x416x3
- Backbone: Darknet-53
- Output Layers: Multiple detection heads

## Performance Metrics
- **Inference Speed**: ~30 FPS
- **mAP (mean Average Precision)**: 0.85
- **False Positive Rate**: <1%

## Project Structure
```ai-weapon-detection/
├── main.py           # Main AI detection script
├── weapons.cfg       # Model configuration
├── weapon.weights    # AI model weights
├── alarm.mp3        # Alert sound
└── requirements.txt  # Dependencies
```

## Contributing
1. Fork the repository
2. Create your feature branch (```git checkout -b feature/AmazingFeature```)
3. Commit your changes (```git commit -m 'Add some AmazingFeature'```)
4. Push to the branch (```git push origin feature/AmazingFeature```)
5. Open a Pull Request


## Contact
Your Name - @aayush-ojha 
Email - aayush.ojha.dev@gmail.com
