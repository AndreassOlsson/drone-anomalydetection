# drone-anomalydetection
This project is about anomaly detection on drone footage and here I will describe the method and progress so far. The final implementation is underway and the simplified versions of the problem show promising results. Note that this repo only contains the ipynb files and is (at the moment) only for demonstration.

The method I am working on implementing utilizes a vision-transformer-encoder and a convolutional decoder. It tries to predict the next frame of a video and then compare the predicted frame with the observed frame. The comparison will be l2 distance. For testing, the model will classify a frame as normal or not based on how big the l2 distance between the prediction and the observation is. This method is based mainly on P. Jin, L. Mou, G. -S. Xia and X. X. Zhu, "Anomaly Detection in Aerial Videos With Transformers," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-13, 2022, Art no. 5628213, doi: 10.1109/TGRS.2022.3198130. The method visualization from that paper is shown below and closely resembles my intended method.
https://arxiv.org/pdf/2209.13363.pdf

![image](https://user-images.githubusercontent.com/59232492/207018544-3de91092-eecb-4e55-9d8b-bc5599805117.png)


A visualization on observation vs prediction is shown below. The image is taken from Avola, Danilo & Cannistraci, Irene & Cascio, Marco & Cinque, Luigi & Diko, Anxhelo & Fagioli, Alessio & Foresti, Gian & Lanzino, Romeo & Mancini, Maurizio & Mecca, Alessio & Pannone, Daniele. (2022). A Novel GAN-Based Anomaly Detection and Localization Method for Aerial Video Surveillance at Low Altitude. Remote Sensing. 14. 4110. 10.3390/rs14164110. https://www.researchgate.net/publication/362870834_A_Novel_GAN-Based_Anomaly_Detection_and_Localization_Method_for_Aerial_Video_Surveillance_at_Low_Altitude

![image](https://user-images.githubusercontent.com/59232492/207019387-bd8a7886-880b-4e1e-a64e-08b5af4aec20.png)


Here, the right left image is the observed "next frame" and the right image is the predicted "next image" and is generated from a number of input frames. This would be considered an anomaly because the prediction differs alot from the observation - the l2 distance would be relatively large. 

In my implementation I started of by creating a visiontransformer mnist classifier just to get the visiontransformer up and running. Then I modified that same classifier to output a latent feature and appended a convolutional decoder. This created an autoencoder which takes an input mnist image, compresses it and then reconstructs it. This applies to anomaly detection in the sence that "normal" images that the model was trained on will be reconstructed succesfully - as opposed to anomalies. 

![image](https://user-images.githubusercontent.com/59232492/207013661-055a0a20-8413-4b10-a91a-e5a8432b7085.png)


The next step was modifying this model to accept sequences of images or "videos". In this step, input data consisted of 6 input frames forming a normal pattern (in this case, an increament of 1) and the 7th frame as the target. The model did not achieve that great results on this task (as shown below) but I still remain optimistic about the implementation on the real dataset. That is because it is not obvious that for example a 4 follows a 3 or that a 8 follows a 7 - just by looking at the previous numbers. There is no real visual pattern here unlike in the real dataset. This may be why my model achieved better results on this particular problem with a tiny model size rather than a regular sized model. Since this problem can be solved just by looking at the last image in the sequence of input frames, a tiny model which only does this may perform better than a bigger model. Looking at the past frames (the temporal aspect) will probably only start being useful when the problem actually requires it, e.g. for when the next frame actually depends on the previous ones. In conclusion, the fact that the model performed poorly on this task does not necessarily mean that it will perform poorly on the real dataset. This step provided useful insight and prepared the model for video input.

![image](https://user-images.githubusercontent.com/59232492/207014374-db7f326b-0550-4e21-8ac9-3370e044e50b.png)


I just finished the dataloaders for the real dataset. Here are 2 sample items from the dataloaders. They look almost the same but this is because they are only 1 frame from each other. If this does not capture the temporal aspect good enough with regards to the speed of which the drone is flying, the intervals may be increased.

![image](https://user-images.githubusercontent.com/59232492/207338706-a41e0ad5-18ef-4012-9ebd-fa6df5d1fe35.png)

Update:
I have implemented the model on the real dataset now. I started by training the model as it was (not very good as it would turn out). Then I changed it and it looked more promising but due to colab's GPU restriction I did not have time to train the updated model for more than one epoch. However, I have access to very powerful GPU's via a webservice and I am basically done with the preparations for using those! I am now very optmistic and believe that I will achieve good results with these fixes!
