def plot_classmap(VGGCAM_weight_path, trainedModel,img_path, label,
nb_classes, num_input_channels, ratio=16):
    """
    Plot class activation map of trained VGGCAM model
    args: VGGCAM_weight_path (str) path to trained keras VGGCAM weights
    img_path (str) path to the image for which we get the activation map
    label (int) label (0 to nb_classes-1) of the class activation map to plot
    nb_classes (int) number of classes
    num_input_channels (int) number of conv filters to add
    in before the GAP layer
    ratio (int) upsampling ratio (16 * 14 = 224)
    """
    # Load and compile model
    modelCAM = VGGCAM(nb_classes, num_input_channels)
    modelCAM.load_weights(VGGCAM_weight_path)
    modelCAM.compile(loss="categorical_crossentropy", optimizer="sgd")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    #vgg model is used to predict class
    label = trainedModel.predict_classes(x.reshape(1, 3, 224, 224),verbose=0)
    batch_size = 1
    classmap = get_classmap(modelCAM,x.reshape(1, 3, 224, 224),nb_classes,batch_size,num_input_channels=num_input_channels,ratio=ratio)
    classes = ["jeans","tshirt"]
    print "PREDICTED LABEL : ", classes[label[0]]
    plt.imshow(img)
    activation = classmap[0,0, :, :]+classmap[0,1, :, :]
    plt.imshow(activation,
    cmap='jet',
    alpha=0.5,
    interpolation='nearest')
    plt.show()
