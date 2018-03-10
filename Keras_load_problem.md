# Keras 模型载入耗时逐渐增加的问题与应对策略
keras多次载入模型  
~~~
keras.models.load_model()  
~~~
其耗时是逐渐增长的  
可能是没有close掉什么文件  
  
具体来说，模型的加载应该分为两部分  
- 模型结构的加载
- 模型参数的加载  

其中模型结构的加载可能需要创建graph  
可能是导致加载耗时逐渐增加的原因  

keras中另提供了一种比较复杂的模型保存、加载方式： 

- 保存部分:
~~~ 
model.save_weights('model.h5')
model_json = model.to_json()
with open('model.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()
~~~
- 加载部分:   
~~~ 
from keras.models import model_from_json
json_file = open("model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
~~~

测试发现，每次加载模型时把模型结构和模型数据均读入时，耗时也是逐渐增多的  
但是，如果模型结构没有发生变化，每次加载只加载模型参数时，耗时可以保持不变

实际应用中，应该不会遇到太多需要不断更改模型结构的情景

我们在使用加载模型的时候，需要注意是否有更新模型结构的需求，如果没有，改成更新模型参数的模式即可

这样的话，新的模型加载模式变成:

- 初始化模型： 
~~~ 
def initialModel():
    if(os.path.exists('model.h5') and os.path.exists('model.json')):
        json_file = open("model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.h5")
    else:
        model = Sequential()
        model.add(Dense(64, input_dim=2))
        model.add(Activation('relu'))
        model.add(Dense(actionNum))
        model.add(Activation('sigmoid'))
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='binary_crossentropy', optimizer=sgd)
        stateInitial=np.random.rand(100,2)
        actionInitial=np.zeros((100,1),dtype=float)
        model.fit(stateInitial,actionInitial, batch_size=100, epochs=100, validation_split=0.1)
        model.save_weights('model.h5')
        model_json = model.to_json()
        with open('model.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close()

    return model
~~~

- 保存模型：
~~~ 
def saveModel(model):
    model.save_weights('model.h5')
~~~ 

- 加载模型：
~~~ 
def loadModel(model):
    model.load_weights("model.h5")
    return model
~~~ 

    
