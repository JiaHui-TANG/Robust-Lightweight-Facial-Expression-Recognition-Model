<h1>Robust Lightweight Facial Expression Recognition Model</h1>
<img src="https://github.com/JiaHui-TANG/Robust-Lightweight-Facial-Expression-Recognition-Model/blob/db602205948bddd84c0268ec7feb9192c4034b21/relevant%20info/nature-of-facial-expression-in-the-wild.png" style="width:600px;height:550px;">

<h2>Description</h2>
<p>Facial expression is a phenomenological response manifested by humans in complex patterns crucial for non-verbal communication. 
  Existing facial expression recognition (FER) networks trained on abundant data have achieved notable gains in accuracy for expression classification. 
  These networks are generally large, with excessive auxiliary components introduced to overcome FER challenges, including occlusions and dynamic head pose variation. 
  Hence, the accuracy improvements depend on the availability of large computational resources. Moreover, existing neural network-based FER models are prone to memorizing training 
  datasets and thus susceptible to corrupted labels and model overfitting. Since facial expression datasets are sensitive information, memorizing these data also indicates critical
  privacy risks.
</p>
  
<h2>Project Objective</h2>
<p>Proposed a robust, lightweight FER model – GhostFace, that leverages depthwise convolution and triplet attention for facial expression classification in the wild to reduce computational cost. This is followed by implementing data augmentation techniques – RandAug and MixUp to enhance the robustness of GhostFace against corrupted data labels and mitigate privacy risks. Experiments conducted on the RAF-DB dataset show the proposed framework achieves promising real-world facial expression classification with a 12.66% reduction in FLOPs. Moreover, incorporating data augmentation can reduce sensitivity to corrupted labels and memorization of sensitive data.</p>

<h2>Data Source</h2>
<p>Real-world Affective Faces Database (RAF-DB) is a large-scale facial expression dataset containing 29672 real-world images, with seven classes of basic emotions: happiness, sadness, anger, fear, surprise, disgust, and neutral. It is more diverse in comparison 
  to the lab-controlled datasets, such as CK+ and MMI that are controlled by psychologists. Subject in the images varies in terms of facial expressions, ethnicity, age group, head poses, illumination conditions, occlusions, and post-processing operations
  (e.g. various filters and special effects). Similar to the existing approach, in this work, we only use 12,271 training set images and 3, 068 test set images for experiments.
      
<h2>Python Packages Used</h2>
<ul>
  <li>retinaface</li>
  <li>tensorflow</li>
  <li>imgaug</li>
</ul>

<h2>Approaches</h2>
<ol>
  <li>Retinaface for image cropping, Image resizing to 224X224, Image normalization</li>
  <li>Visualization of different real-world facial emotions in the dataset</li>
  <li>Developed the <a href=https://github.com/JiaHui-TANG/Robust-Lightweight-Facial-Expression-Recognition-Model/blob/db602205948bddd84c0268ec7feb9192c4034b21/robust_lightweight_FER.py">lightweight robust FER</a> with Tensorflow, evaluate its total parameters, Flops and accuracy score, and save model for model serving</li>
  <ul>
    <li>This step is repeated for the dataset applied with MixUp, RandAug, and a combination of both</li>
    <li>Upon the data augmentation steps, its influence on privacy risks (environmental setting of membership inference attack) is studied and evaluated by their False Positive and False Negative.
  </ul>  
  <li>Plot the model's prediction and area of focus during prediction via GradCam</li>
</ol>
<h2>Deployment and Demonstration</h2>
<a href=https://huggingface.co/spaces/jia-hui-tang/Lightweight_Facial_Expression_Recognition>Click here to use the FER model!</a>
