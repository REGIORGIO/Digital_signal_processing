from classificatorGLCM_NN import convert_images_to_texture_characteristics, get_accuracy

if __name__ == "__main__":
    class_names = ['1', '2', '3', '6', '8']
    root_folder = './wheat/'
    stat_count = 1

    SKO = 0.1
    iteration_count = 10
    averaged_elements = [10]

    input_train_params = root_folder + 'input_train_params.csv'
    input_test_params = root_folder + 'input_test_params.csv'

    convert_images_to_texture_characteristics(root_folder + 'train/', input_train_params)
    convert_images_to_texture_characteristics(root_folder + 'test/', input_test_params)

    get_accuracy(class_names, root_folder, stat_count,
                 input_train_params, input_test_params,
                 SKO, iteration_count, averaged_elements)

