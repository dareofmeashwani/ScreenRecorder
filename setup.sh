pip3 install virtualenv
python3 -m virtualenv -p python3.11 venv
source venv/bin/activate
pip3 install -r requirements-dev.txt
pip3 install -r requirements.txt
brew install python-tk
brew install ffmpeg
brew install loopback