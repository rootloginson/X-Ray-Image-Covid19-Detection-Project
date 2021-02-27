## Covid19 Pneumonia Detection Project by using *Artificial Intelligence* Techniques

Aim of this study is designing a deep learning model to detect Covid 19 disease using x-ray lung images from 
[“Kaggle, CoronaHack -Chest X-Ray-Dataset”](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/metadata) dataset. 
This dataset has unlabeled CT images and labeled x-ray images. 
There are 7 different categories in total, 1 normal and 6 diseases with lung inflammation symptoms. 
Kaggle dataset was highly unbalanced. 
Therefore,  more study has been made on the efficient use of dataset that is detected to have faulty labels compared to the network architecture.

*Personal quote:*  
*Main purpose of being insistent on this dataset is being able to build a good model due to lack of good labeled datas. 
I believe it is very hard to achieve. Many failed attemps help me to understand AI concepts better. 
And gives birth to new ideas* **(っ◔◡◔)っ :sparkles: \\(^.^)/** 

There is a python script in [**this**](https://github.com/rootloginson/Covid-Xray-Eval-Module) repository. 
You can test a lung x-ray image with one line of shell command.

---

<p>&nbsp;</p>

### Kaggle, CoronaHack X-Ray-Dataset Distribution

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
    
- Model type, Training,validation,test accuracy&loss [**graphs**](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/KaggleCovid/Plots). 
- Model type, Training,validation,test accuracy&loss history datas stored in [**.json files**](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/KaggleCovid/tempSave/ModelHistory).
- **Comparisons** tables for different models.

This saved files can be found in this repository along with their **"aboutFiles.txt"** files.

There are 3 colab notebooks along with their detailed explanations.
- [Main Notebok](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Main%20Notebook)

- [Analysing the Coronahack data and creating custom csv for custom dataloader of pytorch](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Data%20Analysis%20CoronaHack%20and%20Making%20Custom%20CSV)

- [Creating custom csv out of torchvision.datasets.ImageFolder type dataset](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Make%20Custom%20CSV%20for%20Test2%20and%20Test3%20Images)
There is also Python Script to create CSV file with one line of shell command.
<<[Script](https://github.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/tree/master/Final/Image_Folder_to_Custom_CSV)>>

<p>&nbsp;</p>

**Model architecture of the succesfull(relatively) model:**

> **Restnet18 (**Pretrained=<span style="color:blue">True</span>**)** 
>>FC (512, 128) >> resnet output << 
>>>Dropout (0.5) 
>>>> FC (128,3)
>>>>> CrossEntropyLoss (log_softmax+nllloss)

<img src="https://raw.githubusercontent.com/rootloginson/X-Ray-Image-Covid19-Detection-Project/master/markdown_files/model_architecture_whitebackground.png" width="800" style="background-color: transparent;">


