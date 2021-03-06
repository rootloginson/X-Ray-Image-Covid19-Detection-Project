## Covid19 Pneumonia Detection Project by using *Artificial Intelligence* Techniques

Aim of this study is designing a deep learning model to detect Covid 19 disease using x-ray lung images from 
[“Kaggle, CoronaHack -Chest X-Ray-Dataset”](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/metadata) dataset. 
This dataset has unlabeled CT images and labeled x-ray images. 
There are 7 different categories in total, 1 normal and 6 diseases with lung inflammation symptoms. 
Kaggle dataset was highly unbalanced. 
Therefore,  more study has been made on the efficient use of dataset that is detected to have faulty labels compared to the network architecture.

There is a python script in [**this**](https://github.com/rootloginson/Covid-Xray-Eval-Module) repository. 
You can test a lung x-ray image with one line of shell command.

---

Research paper ***update***:  
At [this study](https://arxiv.org/pdf/2101.06871.pdf) made by Andrew Y. Ng et al., they have tested the performances of pretrained models that trained on imagenet dataset with chest x-ray images. It can be seen that my Resnet18 result is parallel with the study. 

However, inception v3 assumption was wrong. Inception v3 assumption couldn't be tested due to the lack of computing power. In addition, the insufficiency of the data set and the amount of noise affect this assumption. 

---
<p>&nbsp;</p>

### Kaggle, CoronaHack X-Ray-Dataset Distribution, *Table 1*

|Category|Number of Images|
|:----------|:--------:|
|Normal     |1342|
|Bacteria   |2530|
|Virus      |1345|
|Covid19|58 |58  |
|Sars, Virus|5   |
|Streptococcus Bacteria|4|
|Stress Smoking|2|

<p>&nbsp;</p>

### Neural Network Model

Study has been made for practising. Therefore, there are many scripted process in order to create
    
- Model type, training,validation,test accuracy&loss [**graphs**](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/KaggleCovid/Plots). 
- Model type, training,validation,test accuracy&loss history datas stored in [**.json files**](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/KaggleCovid/tempSave/ModelHistory).
- **Comparisons** tables for different models.
- Final model [state.dict()](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/blob/master/Final/KaggleCovid/tempSave/model2.zip), [Entire Model](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/blob/master/Final/KaggleCovid/tempSave/model2itself.zip)

This saved files can be found in this repository along with their **"aboutFiles.txt"** files.

There are 3 colab notebooks along with their detailed explanations.
- [Main Notebok](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Main%20Notebook)

- [Analysing the Coronahack data and creating custom csv for custom dataloader of pytorch](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Data%20Analysis%20CoronaHack%20and%20Making%20Custom%20CSV)

- [Creating custom csv out of torchvision.datasets.ImageFolder type dataset](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Make%20Custom%20CSV%20for%20Test2%20and%20Test3%20Images)
There is also Python Script to create CSV file with one line of shell command.
<<[Script](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Image_Folder_to_Custom_CSV)>>

<p>&nbsp;</p>

**Modified Resnet18 Model Architecture:** Model architecture of the succesfull(relatively) model


> **Resnet18 (**Pretrained=<span style="color:blue">True</span>**)** 
>>FC (512, 128) >> resnet output << 
>>>Dropout (0.5) 
>>>> FC (128,3)
>>>>> CrossEntropyLoss (log_softmax+nllloss)

<p>&nbsp;</p>


<img src="https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_architecture_whitebackground.png" width="800" style="background-color: transparent;">

**Summary of the models**

After selecting the model structure, different training methods has been used in different combination in order to see the effect of the methods. 
Resnet 50 and Resnet 18 architectetures has been used for pretrained model. These models has been tested along with modified resnet18 model which has been mention above. Every model runs with same settings. Only the output class number has been vary either 3 or 7.
**Result Model** , *unlike others*, has been babysitted to test my understanding about the relation between gradient updates, learning rate, sample distribution. I  led the "Modified Resnet18" model (Result Model) to have a relatively better generalized model compare to others. The algorithm I follow will be explained.

All models except "Result Model" have been trained with the same parameters. These are: 

- All of the training data feeded into model with the *batch_size = 128*
- Training / Validation split is 0.85/0.15
- Training / Validation split is stratified according to training set
- Batches were shuffled each epoch.
- Optimizer method: Stochastic Gradient Descent with Momentum    
- Criterion: torch.nn.CrossEntropyLoss (nn.LogSoftmax + nn.NLLLoss)


```python
optimizer = optim.SGD(params=model.parameters(), 
				      lr=0.001, 
				      momentum=0.95, 
				      weight_decay=0.0002
                      )
 
criterion = nn.CrossEntropyLoss()
```

<p>&nbsp;</p>

As seen in table image below,

Different pretrained Resnet50 and Resnet18 and Modified Resnet18 has been trained **11 epochs** with 7 classes and 3 class outputs.

- 7 classes: Normal, Bacteria, Virus, Covid19, SarsVirus, Strept. Bact., Stress Smoking

- 3 classes: Normal, *Remaining İnflammations such as Bact, Virus, Stress Smoking ...* , Covid19 

Purple colered cells belongs to [“Kaggle, CoronaHack -Chest X-Ray-Dataset”](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/metadata) 's Train and test set. (Note: 3 random Covid images moved into test set from trian set. Which is called **Test**. Light Blue colored cells belongs to [COVID-19 & Normal Posteroanterior(PA) X-rays](https://www.kaggle.com/tarandeep97/covid19-normal-posteroanteriorpa-xrays). Whole dataset used as **Test2**. This dataset contains 140 Normal and 140 Covid Chest X-xray images. Considering the Coronahack dataset has faulty labeled images, test2 dataset was used for comparison. **Test2** dataset was used for comparison purposes after all models including Result Model were trained and the project was terminated. Test2 was not involved in the decision-making process.

***Accuracy and Loss Metric:***  
*Kaggle Coronahack dataset is very small relative to the ImageNet dataset that Resnet had been trained with.  Since its small, to be able to track and understand the dataset easily, arithmetic mean has been used for the training, validation and test set.*
**Confusion matrix of the test results have been created in order to track F1 score.**

Loss: Each epoch,  Arithmetic mean of the Batch Losses

Accuracy: Each epoch, (number of Correct Prediction) / (number of Total Prediction).

 
[![](https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_result_table.png)](https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_result_table.png)

**Interpretation of the results**

1. None of these model except the babysitted "result model" have been able to detect 3 covid images in the test set. But Resnet50, **deeper model**, detected some of the covid images in the Test2 dataset unlike others. In the table image this is denoted by (1) and (2) in orange color. And the models that couldn't detect c ovid images are denoted by the red asterix.

2. Before training all these models, changing class weight of Resnet50 resulted in a same situation where model was able to detect some covid images correctly. This led me to a point I can simulate similar results by changing sample size an distribution.

3. During the training of Result Model, learning rate and sample distribution has been actively changed  depending on the loss, accuracy of training and test dataset, overfit, plateau relationship. The momentum value has also been changed depending on the batch size and sample distribution  either 0.90(last 10), 0.95(last 20). Model detected 3 out of 3 covid images in Test set. This is denoted by the blue asterix.

As a result of this micromanaged experimental work of mine, I believe I have seen the light of why the SOTA applications mentioned in the Stanford Courses and Archive articles have evolved into current network structures. 

Also I wanted to formulize what i did for Result Model (item 4). What I did by hand for changing sample size due to metrics, could have been achieved with Class Weights. This is tested. Maybe? this could give a better solution with "Modified Resnet 50" model. And desicion process to change the learning rate between 0.1 and 0.001 in order to find completely different local minimum can be implemented into training as below algorithm.

Models trained only 11 epochs with a small training set. Looking at the [**graphics**](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/KaggleCovid/Plots), the deeper Resnet model does not seem to cause any significant improvement.

<p>&nbsp;</p>

**Training algorithm of Result Model**

*Start*

>**Train model**  
(if model improves save the model)

*If improves keep training.*

>If F1 score and accuracy dont increase after few epochs, decrease the learning rate.  
Train the model  
(if model improves save the model)

*If improves keep training.*

>If the model is stuck with a poor F1 score and accuracy, change the sample distribution in favor of incorrectly predicted labels.  
Train the model  
(if model improves save model)

*If improves keep training.*

>If the model overfits and metrics drop then it is a bad local minimum.
Go back to last saved model. Increase the learning rate between 10x ~ 100x.  
Probably, gradients will jump into a different local minimum.

> **Back to top.**

*End*

<p>&nbsp;</p>

As a result, 

I have observed that papers which has published by known publisher has quality datas. Medical imaging has done by more advanced machines compare to public datas. And most of the times these datas are not public. Also private hospitals, institution don't share their datas. Few institutions share their datas and these datas are used by researchers. An assumption has been made on that the imaging machines variety (maybe operators?) was not sufficient to generate an acceptably unbiased data set. Also since machine can't be moved, images may carry local variances and similarities like genetics, life styles. For example, if there is industrial air pollution in the area where the tests are performed, or if there is a coal-fired thermoelectric power plant nearby, the normal classified x-ray images will contain lung inflammation [ref](https://pubmed.ncbi.nlm.nih.gov/21623970/).

Converting from RGB to Gray was a experimental idea. Computational power wasn't enough.  Also, practice was done for algorithm and data structure skills.. Therefore some of the codes might seem unnecessary due to that practice reason.

Reducing 7 classes to 3 classes seems usefull but 11 epoch is not good idea to build some intiution on top of it. Computation power wasn't enough and datas were migrated when usage limit exceeded. 

Inflammation is an immune response and it alone is not a unique indicator for covid19 detection. Other type of tests like blood minerals, hormons, hemograms can be taken into consideration in order to build a scalable, usefull prediction model for the future of the disease diagnosis since representative word "allostasis" take care of the body regulation. ([*allostasis*](https://en.wikipedia.org/wiki/Allostasis#cite_note-SterlingEyer1988-2): *remaining stable by being variable*).
For covid detection task, dog nose seems like a better option. Perhaps the combination of [biohybrid nose](https://singularityhub.com/2021/01/26/scientists-made-a-biohybrid-nose-using-cells-from-mosquitoes/) and [AI computer chips](https://interestingengineering.com/ai-computer-chip-smells-danger-could-replace-sniffer-dogs) will be in use in the near future.

Using supervised learning to detect diseases with similar symptoms may not be an efficient idea. Perhaps in the near future, self-supervised learning models will be trained on detailed human body models such as drones trained in a computer game environment. And these models can be combined with AI computer chips that will be implanted in the human body to monitor.

For the visual and auditory intuition from childhood:  
***References: [link1](https://www.youtube.com/watch?v=7AWfzy7wdv4), [link2](https://www.youtube.com/watch?v=cQUf7xma46o)***

Improvements:

There are publications that apply the SVM method for the output of the network and have been shown to be successful.

Personal quote: One of my concerns and my solution suggestion;  
I didn't give enough option to my network to classify images. Network might say this is normal or this is covid. When it is not covid or normal it results as Virus & Bacteria. However, observing the result of the whole sample space of classes would not be practical for the HITL method used in this research. 
I am very curious about adding a class which is not a x-ray image but a gray image with random lines and circles to generalize the model.

<p>&nbsp;</p>

---
**Abstract**

Aim of this study is designing a deep learning model to detect Covid 19 disease using x-ray lung images from “Kaggle, CoronaHack -Chest X-Ray-Dataset” dataset. This dataset has unlabeled CT images and labeled x-ray images . There are 7 different categories in total, 1 normal and 6 disease with lung inflammation symptoms. Kaggle dataset was highly unbalanced. Therefore,  more study has been made on the efficient use of dataset that is detected to have faulty labels compared to the network architecture. For the neural network, FC Layer, Dropout, FC layer on top of pretrained Resnet18 has been used. By using different type of methods on dataset and babysitting the training process; 83% accuracy has been achieved on CoronaHack Test dataset. Observing the behaviour of the training, validation and test set results, a method was followed. Sample sizes of classes and randomly picked images has been changed during the training according to metrics. When the model found a local minimum that could not be generalized for covid19 detection, the learning rate was increased up to 0.1 to find another local optima. When metrics did not improve due to the learning rate, the images were resampled and the sample sizes of each class were changed depends on the situation. To check the applicability of sampling method, class weights of the model has been changed in favor of covid class. With these settings, model was able to predict covid19 images succesfully. Therefore sampling method was applied for practising and checking the intuition about deep learning methods. On two different “covid/normal” test datasets 91% and 88% accuracy has been achieved. About 90% portion of the wrong predictions belong to a "viral or bacterial inflammation" class while ground truth label was covid19. Number of False Negative is 1 out of 140. Number of False Positive is 2 out of 140. 21 out of 140 Covid19 labeled data predicted as Viral&Bacterial inflammation. Datasets were small and noisy. Therefore model cannot be generalized in its current form.

The motivation carried in this project was being able to merge and apply the knowledge that has been learnt during Covid pandemic  o_o

**1 Introduction**

Covid 19 is classified as a viral infection. In symptomatic and severe cases, it results with an inflammation in the lungs . Inflamation is a immune response of a mamal body in order to protect the damaged body area. This inflamation can be observed as fluid increasement. Covid 19-induced increased fluid can be detected on chest x-ray images as an white foggy area on lungs. Human eye can distinguish lung and other body part on xray image in normal cases. During inflamation dark colored lung area can be partly seen. As the fluid increase, white foggy area on x-ray image increases. This white cloud is a common symptom which is observed in other lung diseases as well. Therefore, assumption has been made that using inflammation only as a distinguishing feature for covid and classifying all diseases separately would not significantly increase the success rate of. For this reason, the classification was divided into 3 different groups as Normal, Viral or Bacterial Inflammation and Covid19 Inflamation. This assumption has not been thoroughly tested due to lack of computation power. For computation Google Colab and free provided Google Colab Nvidia T4/16Gb GPU has been used. 

**2 Dataset**

Kaggle Coronahack Chest X Ray dataset disease categories can be seen on Table 1.  Dataset after dividing into  3 different group can be seen on Table 2. TRAIN and Test seperation is belong to Kaggle dataset. However, there was no covid image in the test file. For this reason 3 randomly selected Covid images was moved from training to test folder. 

(*Very bad datas to start with :/*)

CoronaHack **Training** Datas *(Table2)*

|**Category**|**Symptom**|**Class Number**|**Quantity**|
|:------------------|:----------:|:---------:|:-------:|
|Normal                   |Normal      |0          |1342 |
|Others                   |Inflamation |(1,2,4,5,6)|3886 |
|Covid19                  |Inflamation |3          |55   |

CoronaHack **Test** Datas *(Table2)*

|**Category**|**Symptom**|**Class Number**|**Quantity**|
|:------------------|:----------:|:---------:|:-------:|
|Normal                   |Normal      |0          |234 |
|Others                   |Inflamation |(1,2,4,5,6)|390 |
|Covid19                  |Inflamation |3          |3   |


Network training ***without changing class weights or changing sample sizes*** does not result in a situation where Covid 19 images are correctly predicted with this dataset. Therefore Model babysitted. 


**2.1 Custom Dataset and Transformations**

For kaggle dataset, csv file has been modified for custom data loader which has been defined for the pytorch dataloader.  All these intermediate processes can be found on google colab notebook. (These are CustomDatasetRGB function and  processed_dataset.csv file).

**2.2 Color channels**

X-ray images in the Kaggle dataset have Gray, RGB and RGB-A channels.  For the backbone of the model, Pretrained Resnet18 has been used. Input of Resnet18 has 3 channel, 224x224 Input size. Input shape is 3x224x224. All the image channels has been listed. And result showed that majority of the Covid19 files have **RGB** channels and almost all of the Normal datas have **Gray** channel.

Learning model could learn that RGB images are belong to Covid19 class and Gray images are  belong to Normal class. For this reason and to prevent biases caused by channel dimensions and colors, all non-gray images converted into gray images. For this task “convert” function from “Pillow” library has been used (eq.(1)). After this process, to be able to train our Resnet based model, all images converted to RGB with “convert“ function from “Pillow” library (eq.(2)).

|function|Eq|
|--------------------|--:|
|Image.convert(‘L’)  |(1)|
|Image.convert(‘RGB’)|(2)|

**2.3 Horizontal flip**

To be able to use images as a Resnet input; all images resized to 224 pixels with pytorch “Resize” transformation and longer axis has cropped with “CenterCrop” transformation. In order to prevent the biases caused by image position the horizantal flip transform was applied to the image as applied in the Resnet article with the probability of 0.5. 



**2.4 Data normalization parameters**

Images in datasets have different types of attributes such as color and contrast. In order to prevent these differentiation between images, after converting all the images to Grayscale normalization has been applied to the transformed images. Mean (mu) and Standart Deviation (std) has been calculated by using training images of Kaggle, Chest X-ray dataset (eq.(3), eq.(4)). 

|Mean and Standart Deviation|Eq|
|:--------------------:|--:|
|mean = 0.570406436920166  |(3)|
|standart deviation = 0.177922099828720)|(4)|


If dataloader use PIL images, Pytorch automatically scales the pixel values between 0 and 1. 
Model trained with “PIL.convert()” channel transformations. 
When test set was transformed with torch GrayScale and RGB channel transformations, test accuracy dropped 
to 32 percent from 88 percent for [Test3 dataset](https://www.kaggle.com/nabeelsajid917/covid-19-x-ray-10000-images?select=dataset). 
This can be interpreted as two different libraries use different methods for channel transformations.  



**3 Deciding on model** 

**3.1 Considering normalization parameters**

Standart deviation of kaggle dataset is smaller than the Imagenet standart deviation[ref]. This could be interpreted as Imagenet that has 12million different images compare to same type of kaggle dataset lung X-ray images have more standart deviation. Deep neural network models are able to learn functions with only random initialization at first epoch.  And overfit easily due to their complexity. Large training dataset can prevent this from happening. Therefore pretrained deep learning models taken into consideration due to small data size of Coronahack dataset. Two types of options have been picked. One of them is residual networks and the other is networks that has bottleneck layers like inception network. 

**3.2 Eliminating the networks with bottleneck layers**

According to Inception(v3) article [[ref](https://arxiv.org/abs/1512.00567)], in General Design Principles section, “Avoid representational bottlenecks, especially early in the network[[ref](https://arxiv.org/pdf/1512.00567.pdf)]” and “One should avoid bottlenecks with extreme compression[[ref](https://arxiv.org/pdf/1512.00567.pdf)]” These warnings can be interpreted for the covid19 classification tasks as; CNN’s may learn necessary features for the classification task. But bottleneck layers will add up all the learned features and information will be lost. With small dataset unlike imagenet, network may not learn new meaningfull features from bottleneck layers. Same intuition carried for the pooling layers which will be explained in the next section. In addition vanishing and exploiding gradient may become a problem. For these reasons, residual network becomes a strong tactical option against inception network.


**3.3 Residual Networks**

**3.3.1 Why “Residual Networks” ?**

In kaggle dataset there are lungs x-ray images that are similar on a level, to untrained eyes. They may look same from a distance. Images have small standart deviation.  Transformed images are Gray. Therefore important features to seperate Covid and Normal image might have the similar values after few layers depends on the convolution sizes and pooling layers. Inflammations look like an white cloud on top of the lungs. And after few convolution layers electrots and cables may also look like an inflammation to a network. With these intuiton, assumption of Residual Networks might work for not loosing important distinctive features has been made.


**3.3.2 Residual networks and transfer learning**

The residual layers of resnet will prevent the model from exploding and vanishing gradient results. 
Activation of previous layer will be constantly added to forward layer so that useful features can be learned without losing much information of input. Having a pretrained Residual Network trained with Imagenet hypothetically will lower the chance of overfitting compare to untrained version. Another strong motivation for using transfer learning is Andrew Ng’s quote from Deep Learning Specialization Course on Coursera. *“In all the differen disciplines, in all the different applications of deep learning, I think that computer vision is one where transfer learning is something you should almost always do, unless, you have an exceptionally large dataset to train everything else from scratch, yourself* [[ref](https://youtu.be/FQM13HkEfBk?t=496)]".

**3.3.3 The result of untrained resnet training**

Training of  the Untrained Resnet models resulted with overfitting on first epoch. Model found a local minima and accuracy didn’t improve. Kaggle test set accuracy was very low compare to validation set accuracy . Resnet model is complex enough to initialize with local minima for the Kaggle, CoronaHack Chest X-ray set.

