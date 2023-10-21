# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
import tensorflow as tf
print(tf.__version__)

# %%
img=cv2.imread("IMG_CLASSES/1. Eczema 1677/0_10.jpg")
img=cv2.resize(img,(224,224))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

# %%
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

sdir=r'IMG_CLASSES'
train_df, test_df, valid_df= preprocess(sdir, .9,.05)

# %%
train_df['labels'].value_counts()

# %%
size=1132
lst=list(np.unique(np.array(train_df['labels'])))
final_df=pd.DataFrame(columns=['filepaths','labels'])
for i in range(10):
    temp=train_df.loc[train_df['labels']==lst[i]]
    temp.reset_index()
    final_df=pd.concat([final_df,temp.iloc[:size]],axis=0)
final_df.reset_index()
final_df

# %%
#pretrained model

base_model=tf.keras.applications.efficientnet.EfficientNetB5(
    include_top = False,
    input_shape=(224,224,3),
    weights='imagenet',
    classes=10
)
base_model.trainable=False

# %%
for layer in base_model.layers[:-30]:
    layer.trainable=True

# %%
data_aug=tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
    tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
],name='Data_augmentation_layer')

# %%
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

# %%
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rescale=1./255
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rescale=1./255
)

# %%
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

# %%
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='/kaggle/working/skin_model.h5',
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# %%

# %%
model.summary()

# %%
model.fit(
    train_images,
    validation_data=val_images,
    epochs=10,
    callbacks=[checkpoint]
)

# %%
results = model.evaluate(test_images, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))

# %%
model2 = tf.keras.models.load_model('/kaggle/working/skin_model.h5')
results = model2.evaluate(test_images, verbose=0)
model.save("skin.h5")
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


