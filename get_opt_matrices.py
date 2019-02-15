from collab_filter import *

ratings_matrix_for_web = np.load('ratings_matrix_for_web.npy')

print(ratings_matrix_for_web.shape)
print(ratings_matrix_for_web[0].shape)
print(ratings_matrix_for_web[1:,:].shape)


a = predict_by_mf_online(ratings_matrix_for_web[0], ratings_matrix_for_web[1:,:])