# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:26:37.426769Z","iopub.execute_input":"2023-10-21T17:26:37.427026Z","iopub.status.idle":"2023-10-21T17:26:38.736601Z","shell.execute_reply.started":"2023-10-21T17:26:37.427004Z","shell.execute_reply":"2023-10-21T17:26:38.735533Z"}}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:26:38.738569Z","iopub.execute_input":"2023-10-21T17:26:38.739437Z","iopub.status.idle":"2023-10-21T17:27:45.031775Z","shell.execute_reply.started":"2023-10-21T17:26:38.739410Z","shell.execute_reply":"2023-10-21T17:27:45.030726Z"}}
import tensorflow as tf
print(tf.__version__)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:45.032899Z","iopub.execute_input":"2023-10-21T17:27:45.033529Z","iopub.status.idle":"2023-10-21T17:27:45.506272Z","shell.execute_reply.started":"2023-10-21T17:27:45.033474Z","shell.execute_reply":"2023-10-21T17:27:45.505362Z"}}
img=cv2.imread("/kaggle/input/skin-diseases-image-dataset/IMG_CLASSES/1. Eczema 1677/0_10.jpg")
img=cv2.resize(img,(224,224))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:45.508297Z","iopub.execute_input":"2023-10-21T17:27:45.508611Z","iopub.status.idle":"2023-10-21T17:27:53.098595Z","shell.execute_reply.started":"2023-10-21T17:27:45.508586Z","shell.execute_reply":"2023-10-21T17:27:53.097695Z"}}
def preprocess (sdir, trsplit, vsplit):
    filepaths=[]
    labels=[]    
    classlist=os.listdir(sdir)
    for klass in classlist:
        classpath=os.path.join(sdir,klass)
        flist=os.listdir(classpath)
        for f in flist:
            fpath=os.path.join(classpath,f)
            filepaths.append(fpath)
            labels.append(klass)
    Fseries=pd.Series(filepaths, name='filepaths')
    Lseries=pd.Series(labels, name='labels')
    df=pd.concat([Fseries, Lseries], axis=1)       
    # split df into train_df and test_df 
    dsplit=vsplit/(1-trsplit)
    strat=df['labels']    
    train_df, dummy_df=train_test_split(df, train_size=trsplit, shuffle=True, random_state=123, stratify=strat)
    strat=dummy_df['labels']
    valid_df, test_df=train_test_split(dummy_df, train_size=dsplit, shuffle=True, random_state=123, stratify=strat)
    print('train_df length: ', len(train_df), '  test_df length: ',len(test_df), '  valid_df length: ', len(valid_df))
    print(train_df['labels'].value_counts())
    return train_df, test_df, valid_df

sdir=r'C:/Users/dest4/Desktop/bulutklinik-V1.2/skin_diseases/IMG_CLASSES'
train_df, test_df, valid_df= preprocess(sdir, .9,.05)
    

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:53.099758Z","iopub.execute_input":"2023-10-21T17:27:53.100036Z","iopub.status.idle":"2023-10-21T17:27:53.109681Z","shell.execute_reply.started":"2023-10-21T17:27:53.100012Z","shell.execute_reply":"2023-10-21T17:27:53.108820Z"}}
train_df['labels'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:53.110872Z","iopub.execute_input":"2023-10-21T17:27:53.111228Z","iopub.status.idle":"2023-10-21T17:27:53.199868Z","shell.execute_reply.started":"2023-10-21T17:27:53.111197Z","shell.execute_reply":"2023-10-21T17:27:53.199061Z"}}
size=1132
lst=list(np.unique(np.array(train_df['labels'])))
final_df=pd.DataFrame(columns=['filepaths','labels'])
for i in range(10):
    temp=train_df.loc[train_df['labels']==lst[i]]
    temp.reset_index()
    final_df=pd.concat([final_df,temp.iloc[:size]],axis=0)
final_df.reset_index()
final_df

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:53.200871Z","iopub.execute_input":"2023-10-21T17:27:53.201108Z","iopub.status.idle":"2023-10-21T17:27:59.788251Z","shell.execute_reply.started":"2023-10-21T17:27:53.201087Z","shell.execute_reply":"2023-10-21T17:27:59.787242Z"}}
#pretrained model

base_model=tf.keras.applications.efficientnet.EfficientNetB5(
    include_top = False,
    input_shape=(224,224,3),
    weights='imagenet',
    classes=10
)
base_model.trainable=False

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:59.789505Z","iopub.execute_input":"2023-10-21T17:27:59.789855Z","iopub.status.idle":"2023-10-21T17:27:59.817063Z","shell.execute_reply.started":"2023-10-21T17:27:59.789823Z","shell.execute_reply":"2023-10-21T17:27:59.816104Z"}}
for layer in base_model.layers[:-30]:
    layer.trainable=True

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:59.818260Z","iopub.execute_input":"2023-10-21T17:27:59.818544Z","iopub.status.idle":"2023-10-21T17:27:59.843593Z","shell.execute_reply.started":"2023-10-21T17:27:59.818517Z","shell.execute_reply":"2023-10-21T17:27:59.842823Z"}}
data_aug=tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
    tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
],name='Data_augmentation_layer')

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:27:59.846273Z","iopub.execute_input":"2023-10-21T17:27:59.846548Z","iopub.status.idle":"2023-10-21T17:28:01.452027Z","shell.execute_reply.started":"2023-10-21T17:27:59.846525Z","shell.execute_reply":"2023-10-21T17:28:01.451250Z"}}
#Normal model

inputs=tf.keras.Input(shape=(224,224,3),name="input_layer")
x=data_aug(inputs)
x=base_model(x)
x=tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x=tf.keras.layers.Dense(256,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(l = 0.016),kernel_initializer=tf.keras.initializers.he_normal(),name='dense_layer2')(x)
x=tf.keras.layers.Dropout(0.5,seed=123)(x)
outputs=tf.keras.layers.Dense(10,activation='softmax',dtype=tf.float32)(x)

model=tf.keras.Model(inputs,outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy','Precision']
             )



# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:01.453094Z","iopub.execute_input":"2023-10-21T17:28:01.453378Z","iopub.status.idle":"2023-10-21T17:28:01.458357Z","shell.execute_reply.started":"2023-10-21T17:28:01.453354Z","shell.execute_reply":"2023-10-21T17:28:01.457517Z"}}
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rescale=1./255
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rescale=1./255
)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:01.459474Z","iopub.execute_input":"2023-10-21T17:28:01.459790Z","iopub.status.idle":"2023-10-21T17:28:30.460028Z","shell.execute_reply.started":"2023-10-21T17:28:01.459767Z","shell.execute_reply":"2023-10-21T17:28:30.459035Z"}}
train_images = train_generator.flow_from_dataframe(
    final_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224,224),
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    batch_size=32,
    seed=42
)

val_images = train_generator.flow_from_dataframe(
    valid_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224,224),
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    batch_size=16
)

test_images = test_generator.flow_from_dataframe(
    test_df,
    x_col='filepaths',
    y_col='labels',
    target_size=(224,224),
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    batch_size=10
)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:30.461201Z","iopub.execute_input":"2023-10-21T17:28:30.462082Z","iopub.status.idle":"2023-10-21T17:28:30.466234Z","shell.execute_reply.started":"2023-10-21T17:28:30.462055Z","shell.execute_reply":"2023-10-21T17:28:30.465248Z"}}
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/kaggle/working/skin_model.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:30.469097Z","iopub.execute_input":"2023-10-21T17:28:30.469372Z","iopub.status.idle":"2023-10-21T17:28:30.649603Z","shell.execute_reply.started":"2023-10-21T17:28:30.469343Z","shell.execute_reply":"2023-10-21T17:28:30.648623Z"}}
tf.keras.utils.plot_model(base_model, show_shapes=True)

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:30.650982Z","iopub.execute_input":"2023-10-21T17:28:30.651302Z","iopub.status.idle":"2023-10-21T17:28:30.714461Z","shell.execute_reply.started":"2023-10-21T17:28:30.651274Z","shell.execute_reply":"2023-10-21T17:28:30.713525Z"}}
base_model.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-10-21T17:28:30.715854Z","iopub.execute_input":"2023-10-21T17:28:30.716165Z"}}
base_model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    callbacks=[checkpoint]
)

# %% [code]
results = model.evaluate(test_images, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# %% [code]
model2 = tf.keras.models.load_model('/kaggle/working/skin_model.h5')
results = model2.evaluate(test_images, verbose=0)
model.save("skin.h5")
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))