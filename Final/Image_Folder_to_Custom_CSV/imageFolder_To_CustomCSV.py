import os
import shutil
import argparse
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser(description= 'Pass file names to coresponding diseas.')
parser.add_argument('-D', '--Test_Directory', type=str, required=True, help='name of the test test directory, (parent dir of the images')
parser.add_argument('-N', '--Normal_Image', type=str, required=True, help='directory name of the Normal Images')
parser.add_argument('-C', '--Covid_Image', type=str, required = True, help='directory name of the Covid Images')
parser.add_argument('-R', '--Remaining_Image', type=str, required=False, help='seperate folder names with single comma')
args = parser.parse_args()

print(args)
current_dir_path = os.getcwd()

# wd is testfile adress
# example ./SomeFolder/AnotherFolder/TestFile
wd = current_dir_path + '/' + args.Test_Directory

# label dictionary = {0:'Normal', 1:'Bacteria,Vırus etc', 2:'Covid19')
path0 = wd + '/' + args.Normal_Image
path1 = wd + '/' + args.Remaining_Image
path2 = wd + '/' + args.Covid_Image

dir_path = [path0, path1, path2]

# create a list of [image name, diseas type] pairs
name_class = [[f,int(ix)] for ix,d in enumerate(dir_path,start=0)
                          for f in os.listdir(d)]

# create pandas dataframe with 'X_ray_image_name' and 'Class'(class is diseas type)
df1 = pd.DataFrame(name_class, columns=['X_ray_image_name', 'Class'])

#print(df1.info(), '\n');
print(df1.Class.value_counts(), '\n');

# copy of the each class to new dataframe to detect image channel numbers
test_normal = df1[df1['Class']==0]
test_rest = df1[df1['Class']==1]
test_covid = df1[df1['Class']==2]



#################################################
####    Count Cahnnel Dimension of Images    ####
#################################################

t_normal = test_normal.copy()
t_rest = test_rest.copy()
t_covid = test_covid.copy()

# extract the chanel dim of each image. create a dictionary for each channel dim

channel_dimension_dict = {}

for ix,t_chdim in enumerate([t_normal, t_rest, t_covid], start=0):
    for i in range(len(t_chdim.index)):
        # corresponding image name
        img_name = t_chdim.iloc[i]['X_ray_image_name']
        # open image
        img = Image.open(dir_path[ix] + '/' + img_name)
        # number of channel that image has
        dim = img.getbands()

        # create key with the dimension if not exist
        if dim not in channel_dimension_dict:
            channel_dimension_dict[dim] = []

        # add image name to corresponding channel dim(dictionary key)
        channel_dimension_dict[dim].append(img_name)

# {dim:[img_name, ...], dim:[img_name] ...}
print("There are>>", channel_dimension_dict.keys(), "< channel Images\n")

# img quantity in channels
dict_info = [(k, len(v)) for k, v in channel_dimension_dict.items()]
print("image quantity of channels")
print(dict_info)
print("\ntotal ", sum(i[1] for i in dict_info))
print("")

# create a dict of{img_name:img_dim}
reversed_channel_dict = {v_in:len(k) for (k,v) in channel_dimension_dict.items() for v_in in v}
# create data set of channel dimensions. row: ImageName, column: Channel_dim
dataset_channel_dim_df = pd.DataFrame.from_dict(reversed_channel_dict, orient='index', columns=['Channel_dim'])

print(dataset_channel_dim_df.value_counts())
print("\n size:",dataset_channel_dim_df.size)

# save the channel dim data as csv
save_path_of_channel_dim = current_dir_path +'/dataset_channel_dim_of_test_file.csv'
dataset_channel_dim_df.to_csv(save_path_of_channel_dim, index_label='X_ray_image_name')


#################################################
#### Adding Channel Dimensions to custom CSV ####
#################################################

# sorted channel dim, index is image name 
channel_dim_list = pd.read_csv(save_path_of_channel_dim, index_col=0).sort_index()

# kaggle_dataset, index = X_ray_image_name, column=Class
#nf = new frame abbrv.
nf_temp = df1[['X_ray_image_name', 'Class']].set_index('X_ray_image_name').sort_index()

nf_temp['Channel_dim'] = channel_dim_list['Channel_dim']

# Save Processed csv.
save_path_of_processed_dataset = current_dir_path + '/processed_dataset_test.csv'
nf_temp.to_csv(save_path_of_processed_dataset, index_label='X_ray_image_name')

#################################################
####    Move images to a single directory    ####
#################################################

# create a new test file directory
moving_dir_name = current_dir_path + '/' + 'useThisFileForTest'
if os.path.exists(moving_dir_name):
    pass
else:
    try:
        os.mkdir(moving_dir_name)
    except:
        print("something wrong.. create target file for copying")

for path in dir_path:
    file_names = os.listdir(path)
    for f in file_names:
        shutil.move(os.path.join(path, f), moving_dir_name)