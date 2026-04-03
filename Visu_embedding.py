import numpy as np

embeddings = np.load('results_test/mobileclip/mobileclip_s2_embeddings.npy')

print(embeddings.shape)       # nombre de frames x dimension des vecteurs, ex : (20, 512)
print(embeddings.dtype)       # type des données, ex : float32
print(embeddings[0])          # le vecteur de la première frame (512 valeurs)
print(embeddings[0][:10])     # juste les 10 premières valeurs pour ne pas tout afficher
print(embeddings.min(), embeddings.max())  # plage de valeurs