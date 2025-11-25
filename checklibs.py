# Import libraries and check if they are installed

libraries = [
    "tensorflow", 
    "keras", 
    "torch", 
    "torchvision", 
    "torchaudio", 
    "numpy", 
    "pandas", 
    "matplotlib", 
    "seaborn", 
    "cv2", 
    "scipy", 
    "scikit_learn"
]

for lib in libraries:
    try:
        globals()[lib] = __import__(lib)
        print(f"{lib} is installed!")
    except ImportError:
        print(f"{lib} is NOT installed.")
