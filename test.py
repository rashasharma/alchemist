import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
perfume_a = np.array([[1, 0]])
perfume_b = np.array([[0, 1]])
perfume_c = np.array([[1, 1]])

print("Similarity between Vanilla and Smoke:", cosine_similarity(perfume_a, perfume_b))
print("Similarity between Vanilla and Mix:", cosine_similarity(perfume_a, perfume_c))