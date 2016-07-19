Implementation of Google Developers' [Train an Image Classifier with TensorFlow for Poets - Machine Learning Recipes #6](https://youtu.be/cSKfRcEDGUs?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal) / [TensorFlow for Poets Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/).

The commands below are slightly modified from the instructions, since I wanted to try doing this without docker.

Image set is `.gitignore`d. To get images:

    mkdir tf_files
    cd tf_files
    curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
    tar xzf flower_photos.tgz
    rm flower_photos.tgz

To just not bother with Docker (also gitignored):

    pip install tensorflow
    git clone https://github.com/tensorflow/tensorflow.git

See [this question on Stack Overflow](http://stackoverflow.com/questions/38218274/tensorflow-importerror-for-graph-util-from-tensorflow-python-framework) and [this issue on the TensorFlow GitHub repo](https://github.com/tensorflow/tensorflow/issues/3203) if you run into an `ImportError` with `graph_util`. I have solved my problem by checking out the r0.9 branch (`git checkout remotes/origin/r0.9
`).

To re-train network (this is the "I'm not in a hurry" option in the instructions, takes a while):

    python tensorflow/tensorflow/examples/image_retraining/retrain.py \
    --bottleneck_dir=tf_files/bottlenecks \
    --model_dir=tf_files/inception \
    --output_graph=tf_files/retrained_graph.pb \
    --output_labels=tf_files/retrained_labels.txt \
    --image_dir tf_files/flower_photos

Try to classify a daisy:

    python label_image.py tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
    # daisy (score = 0.99796)
    # sunflowers (score = 0.00145)
    # dandelion (score = 0.00046)
    # tulips (score = 0.00012)
    # roses (score = 0.00001)

Then a rose:

    python label_image.py tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg
    # roses (score = 0.98240)
    # tulips (score = 0.01685)
    # dandelion (score = 0.00045)
    # sunflowers (score = 0.00023)
    # daisy (score = 0.00007)