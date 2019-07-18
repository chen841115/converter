import sys
import onnx_pb2
import onnx
#import onnx_operators_pb2

model_file = "model.onnx"
model = onnx_pb2.ModelProto()

with open(model_file,"rb") as f:
    model.ParseFromString(f.read())


wf = open("log.txt","wb")

#print(model.ir_version)
#print(model.producer_name)
#print(model.producer_version)
#print(model.domain)
#print(model.model_version)
#print(model.doc_string)
#print("opset")
#for opset in model.opset_import:
#    print(opset.domain)

print("graph")
graph = model.graph
#print(graph.name)
'''
i = 1
print(graph.input[i])
print(graph.node[i].name)
print("node input")
print(graph.node[i].input)
print("node output")
print(graph.node[i].output)
print("node op")
print(graph.node[i].op_type)
print("node attribute")
print(graph.node[i].attribute)
print(graph.output[i])
'''
#print(graph.output)
#print("graph init")
'''
for i in range(0,5):
    print("i = ",i)
    print(graph.initializer[i].raw_data)
'''
#print(graph.input)
print(graph.initializer[4])
batchnor = 0
activation = ""

for node in graph.node:
    if(len(node.output)>1):
        print("\033[0;32;40m")
        print("output > 1")
        print(len(node.output))
        print("\033[0m")
    if(node.op_type == "Conv"):
        size_set = False
        print("\033[0;32;40m")
        print("[convolutional]")
        print("\033[0m")
        #print("node name:",node.name)
        #print("input node name:",node.input)
        if (len(node.input)>1):
            for input in graph.input:
                if(node.input[1] == input.name):
                    print("\033[0;34;40m","filters=%d"%input.type.tensor_type.shape.dim[0].dim_value,"\033[0m") #dim[0]: filters dim[1]:channels dim[2:3]: size
                    print("\033[0;34;40m","size=%d"%input.type.tensor_type.shape.dim[2].dim_value,"\033[0m")
                    size_set = True
        #print("optype:",node.op_type)
        #print("attribute:",node.attribute)
        #print("output node name:",node.output)
        for attri in node.attribute:
            if(attri.name == "strides"):
                print("\033[0;34;40m","stride=%d"%attri.ints[0],"\033[0m")
            elif(attri.name == "kernel_shape" and (not size_set)):
                print("\033[0;34;40m","size=%d"%attri.ints[0],"\033[0m")
            elif(attri.name == "auto_pad"):
                if(attri.s == b'SAME_UPPER'):
                   print("\033[0;34;40m","pad=1","\033[0m")
                elif(attri.s == b'SAME_LOWER'):
                    print("\033[0;34;40m","pad=1","\033[0m")
                elif(attri.s == b'VALID'):
                    print("\033[0;34;40m","pad=0","\033[0m")
                elif(attri.s == b'NOTSET'):
                    print("\033[0;34;40m","pad=0","\033[0m")
            elif(attri.name == "pads"):
                #print("\033[0;34;40m","padding=%d"%attri.ints[0],"\033[0m")
                print("not define yet")
    elif(node.op_type == "MaxPool"):

        print("\033[0;32;40m")
        print("[maxpool]")
        print("\033[0m")
        for attri in node.attribute:
            if(attri.name == "strides"):
                print("\033[0;34;40m","stride=%d"%attri.ints[0],"\033[0m")
            elif(attri.name == "kernel_shape"):
                print("\033[0;34;40m","size=%d"%attri.ints[0],"\033[0m")
#        print(node.attribute)

    elif(node.op_type == "BatchNormalization"):
        print("\033[0;34;40m","batch_normalize=1","\033[0m")
        #print("\033[0;34;40m","Batch","\033[0m")
    elif(node.op_type == "LeakyRelu"):
        print("\033[0;34;40m","activation=leaky","\033[0m")
       # print("\033[0;34;40m","LRelu","\033[0m")
    else:
        print("\033[0;35;40m","Skip ",node.op_type,"\033[0m" )

#print(model.metadata_props)
#node = graph.node[1]

print(graph.node[len(graph.node)-1])
print(graph.output)

model1 = onnx.load("model.onnx")
onnx.helper.printable_graph(model1.graph)
#wf.write(graph.initializer[0].raw_data)
wf.write("123".encode('utf8'))
wf.close()


#print(len(graph.node))
