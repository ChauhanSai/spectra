Hello everyone,

Here is the week3 folder for building your defenses. That is the updated version of the code, and I removed older attack ideas that we are not using anymore.

The main file you will probably work with is week3/src/infer_qwen.py. That file runs the model on the attacked images using the prompt: "What is the main object in this image? Reply with one word only." Right now it uses Qwen, but you can swap the model if you want. Most likely, your defense will go somewhere in this inference pipeline, either directly in that file or by calling another function before the image reaches the model.

I have been testing on 200 sampled images to keep things consistent. The list of image names is in week3/outputs/sample/sampled_images.txt. I already generated and committed the attacked images, so you can use those directly in week3/outputs/attacked/flower and week3/outputs/attacked/knife.

If you want to change the attacks/attacked images themselves, look at week3/src/generate_attacks.py. To rerun attack generation from scratch, you would need the COCO training images in data/coco/train2017 so that generate attacks can obtain the souce images. I did not commit the full dataset because it is too large, but the attacked images that are already in the repo should be enough for implementing and testing defenses.

Sorry about the confusing code, the key thing to remember is that you're probably going to modify infer_qwen, and that 200 attacked images for repeated testing are provided to you.
