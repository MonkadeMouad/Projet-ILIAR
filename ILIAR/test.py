import numpy as np

try:
    # Utilisation de allow_pickle=True pour charger les objets
    data = np.load('/home/profil/monkade1u/ros2_ws/dataset/chunk_0.npz', allow_pickle=True)
    print("Data loaded successfully.")
    arr = np.arange(12).reshape((3, 4))
    # Afficher les clés du fichier .npz
    print("Keys in the dataset:", np.extract(data,arr=arr))
    
    # Afficher le contenu de chaque tableau
    for key in data.files:
        print(f"{key}: {data[key]}")
except Exception as e:
    print(f"Error loading the file: {e}")
