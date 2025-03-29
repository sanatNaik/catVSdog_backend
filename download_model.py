import gdown

url = "https://drive.google.com/file/d/1bhuyFJkOWS8uA9cvE8F6B3TCuYzXnIZ0/view?usp=sharing"
output = "catVdogCNN.h5"
gdown.download(url, output, quiet=False)
