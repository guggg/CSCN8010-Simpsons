import keras
import numpy as np

labels = ['abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson',
          'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 
          'disco_stu', 'edna_krabappel', 'fat_tony', 'gil', 'groundskeeper_willie', 'homer_simpson',
          'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'maggie_simpson',
          'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
          'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier','principal_skinner',
          'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob', 
          'sideshow_mel', 'snake_jailbird', 'troy_mcclure', 'waylon_smithers']

def model_run(model, image):
    if model == 'CNN':
        model = keras.models.load_model('./models/cus.h5')
    elif model == 'MobileNetV2':
        model = keras.models.load_model('./models/MNV2.h5')
    elif model == 'ResNet50':
        model = keras.models.load_model('./models/RN50.h5')
    else:
        print('Invalid model selected. Please choose from CNN, MobileNetV2, or ResNet50.')
        exit()

    image_arr = np.expand_dims(np.array(image), axis=0)

    return labels[np.argmax(model.predict(image_arr)[0], axis=0)]
