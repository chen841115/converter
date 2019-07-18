import onnx
from onnx import helper
import sys,getopt
#from array import array
import struct
#加载模型
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

#获取节点和节点的输入输出名列表，一般节点的输入将来自于上一层的输出放在列表前面，参数放在列表后面
def getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:   #some name wont have value need edit
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node,input_name,output_name

#获取对应输入信息
def getInputTensorValueInfo(input_name,model):
    in_tvi = []
    for name in input_name:
        for params_input in model.graph.input:
            if params_input.name == name:
               in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

#获取对应输出信息
def getOutputTensorValueInfo(output_name,model):
    out_tvi = []
    for name in output_name:
        out_tvi = [inner_output for inner_output in model.graph.value_info if inner_output.name == name]
        if name == model.graph.output[0].name:
            out_tvi.append(model.graph.output[0])
    return out_tvi

#获取对应超参数值
def getInitTensorValue(input_name,model):
    init_t = []
    for name in input_name:
        init_t = [init for init in model.graph.initializer if init.name == name]
    return init_t

#构建单个节点onnx模型
def createSingelOnnxModel(ModelPath,nodename,SaveType="",SavePath=""):
    model = loadOnnxModel(str(ModelPath))
    Node,input_name,output_name = getNodeAndIOname(nodename,model)
    in_tvi = getInputTensorValueInfo(input_name,model)
    out_tvi = getOutputTensorValueInfo(output_name,model)
    init_t = getInitTensorValue(input_name,model)

    graph_def = helper.make_graph(
                [Node],
                nodename,
                inputs=in_tvi,  # 输入
                outputs=out_tvi,  # 输出
                initializer=init_t,  # initalizer
            )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    print(nodename+"onnx模型生成成功！")


wf = open("model.weight","wb")
cf = open("model.cfg","w")
model = loadOnnxModel('./bvlc_alexnet/model.onnx')
#model = loadOnnxModel('./model.onnx')

#print (model.graph)
graph = model.graph
#for node in graph.node:
    #print("node: ",node,"\n")
    #print("----------\n")

# for initializer in graph.initializer:
#     print("data_type: ",initializer.data_type,"\n")
#     print("dims: ",initializer.dims,"\n")
#     print("doc_string: ",initializer.doc_string,"\n")
#     print("double_data: ",initializer.double_data,"\n")
#     print("float_data: ",initializer.float_data,"\n")
#     print("int32_data: ",initializer.int32_data,"\n")
#     print("int64_data: ",initializer.int64_data,"\n")
#     print("name: ",initializer.name,"\n")
#     print("segment: ",initializer.segment,"\n")
#     print("string_data: ",initializer.string_data,"\n")
#     print("----------\n")

# for input in graph.input:
#     print("input: ",input,"\n")
#     print("----------\n")

for input in graph.input:
    print("\033[0;34;40m","[net]","\033[0m\n")
    cf.write("[net]\n")
    print("\033[0;34;40m","height=%d"%input.type.tensor_type.shape.dim[2].dim_value,"\033[0m")
    cf.write("height=%d\n"%input.type.tensor_type.shape.dim[2].dim_value)
    print("\033[0;34;40m","width=%d"%input.type.tensor_type.shape.dim[3].dim_value,"\033[0m")
    cf.write("width=%d\n"%input.type.tensor_type.shape.dim[3].dim_value)
    print("\033[0;34;40m","channels=%d"%input.type.tensor_type.shape.dim[1].dim_value,"\033[0m")
    cf.write("channels=%d\n"%input.type.tensor_type.shape.dim[1].dim_value)
    break
conv_filter = []
L=-1
last=0
for node in graph.node:
    print("op_type",node.op_type)
    #print(node)
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
        cf.write("\n[convolutional]\n")
        #print("node name:",node.name)
        print("input node name:",node.input[1])
        if (len(node.input)>1):
            for gi in graph.initializer:
                if(len(node.input)>2):
                    #get bias float 
                    #write bias float to file
                    continue
                if(gi.name == node.input[1]):
                    if(gi.raw_data):
                        #print(gi.raw_data)
                        continue
                    elif(gi.float_data):  #else int32_data, int64_data, double_data .etc  find at https://hexdocs.pm/onnxs/Onnx.TensorProto.html
                        #print(gi.float_data)
                        #gi.float_data.tofile(wf)
                        wf.write(struct.pack('<%df' % len(gi.float_data), *gi.float_data))
                    break
            for input in graph.input:
                if(node.input[1] == input.name):
                    print("\033[0;34;40m","filters=%d"%input.type.tensor_type.shape.dim[0].dim_value,"\033[0m") #dim[0]: filters dim[1]:channels dim[2:3]: size
                    print("\033[0;34;40m","size=%d"%input.type.tensor_type.shape.dim[2].dim_value,"\033[0m")
                    cf.write("filters=%d\n"%input.type.tensor_type.shape.dim[0].dim_value)
                    cf.write("size=%d\n"%input.type.tensor_type.shape.dim[2].dim_value)
                    size_set = True
                    conv_filter.append(input.type.tensor_type.shape.dim[0].dim_value)
                    print(conv_filter[0])
                    L=L+1

        #print("optype:",node.op_type)
        #print("attribute:",node.attribute)
        #print("output node name:",node.output)
        for attri in node.attribute:
            print("node.attribute",attri)
            if(attri.name == "strides"):
                print("\033[0;34;40m","stride=%d"%attri.ints[0],"\033[0m")
                cf.write("stride=%d\n"%attri.ints[0])
            elif(attri.name == "kernel_shape" and (not size_set)):
                print("\033[0;34;40m","size=%d"%attri.ints[0],"\033[0m")
                cf.write("size=%d\n"%attri.ints[0])
            elif(attri.name == "auto_pad"):
                if(attri.s == b'SAME_UPPER'):
                   print("\033[0;34;40m","pad=1","\033[0m")
                   cf.write("pad=1\n")
                elif(attri.s == b'SAME_LOWER'):
                    print("\033[0;34;40m","pad=1","\033[0m")
                    cf.write("pad=1\n")
                elif(attri.s == b'VALID'):
                    print("\033[0;34;40m","pad=0","\033[0m")
                    cf.write("pad=0\n")
                elif(attri.s == b'NOTSET'):
                    print("\033[0;34;40m","pad=0","\033[0m")
                    cf.write("pad=0\n")
            elif(attri.name == "pads"):
                #print("\033[0;34;40m","padding=%d"%attri.ints[0],"\033[0m")
                #print("not define yet")
                print("\033[0;34;40m","pad=%d"%attri.ints[0],"\033[0m")
                cf.write("pad=%d\n"%attri.ints[0])
            elif(attri.name == "group"):
                if(attri.i!=1):
                    print("\033[0;34;40m","groups=%d"%attri.i,"\033[0m")
                    cf.write("groups=%d\n"%attri.i)
    elif(node.op_type == "MaxPool"):

        print("\033[0;32;40m")
        print("[maxpool]")
        print("\033[0m")
        print(node.attribute)
        cf.write("\n[maxpool]\n")
        for attri in node.attribute:
            if(attri.name == "strides"):
                print("\033[0;34;40m","stride=%d"%attri.ints[0],"\033[0m")
                cf.write("stride=%d\n"%attri.ints[0])
            elif(attri.name == "kernel_shape"):
                print("\033[0;34;40m","size=%d"%attri.ints[0],"\033[0m")
                cf.write("size=%d\n"%attri.ints[0])
#        print(node.attribute)
    elif(node.op_type == "GlobalAveragePool"):
        print("\033[0;32;40m")
        print("[avgpool]")
        print("\033[0m")
        print(node.attribute)
        cf.write("\n[avgpool]\n")

    elif(node.op_type == "Upsample"):
        print("\033[0;32;40m")
        print("[upsample]")
        print("stride=2")
        print("\033[0m")
        cf.write("\n[upsample]\n")
        cf.write("stride=2\n")
        print("node.name",node.name)

    elif(node.op_type == "Concat"):
        print("\033[0;32;40m")
        print("@@@@@@@@@@")
        print("\033[0m")
        for attri in node.input:
            print("node.input",attri)
        

    elif(node.op_type == "Add"):
        #print(node)
        print("\033[0;32;40m")
        print("[ADD]")
        print("\033[0m")
        # print(node.attribute)
        # print(node.op_type.Inputs)
        # print(node.op_type.Outputs)
        cf.write("\n[shortcut]\n")    
        print("L:",L,"\n")
        point=L-1
        while point >= 0:
            print("point:",point,"\n")
            if(conv_filter[L]==conv_filter[point]):
                last=point
                break
            point=point-1

        print("\033[0;31;40m","from=-%d\n"%(L-last+1),"\033[0m")
        cf.write("from=-%d\n"%(L-last+1))
        print("\033[0;31;40m","activation=linear\n","\033[0m")
        cf.write("activation=linear\n")


    elif(node.op_type == "BatchNormalization"):
        print("\033[0;34;40m","batch_normalize=1","\033[0m")
        cf.write("batch_normalize=1\n")
        print(node.attribute)
        #print("\033[0;34;40m","Batch","\033[0m")
    elif(node.op_type == "LeakyRelu"):
        print("\033[0;34;40m","activation=leaky","\033[0m")
        cf.write("activation=leaky\n")
        #print("\033[0;34;40m","LRelu","\033[0m")
    elif(node.op_type == "Relu"):
        print("\033[0;34;40m","activation=relu","\033[0m")
        cf.write("activation=relu\n")
        #print("\033[0;34;40m","relu","\033[0m")
    elif(node.op_type == "Dropout"):
        print("\033[0;32;40m")
        print("[dropout]")
        print("\033[0m")
        cf.write("\n[dropout]\n")
        for attri in node.attribute:
            print("node.attribute",attri)
            if(attri.name == "ratio"):
                #print("\033[0;34;40m","stride=%d"%attri.ints[0],"\033[0m")
                #cf.write("stride=%d\n"%attri.ints[0])
                print("\033[0;34;40m","probability=%f"%attri.f,"\033[0m")
                cf.write("probability=%f\n"%attri.f)
    elif(node.op_type == "Softmax"):
        print("\033[0;32;40m")
        print("[softmax]")
        print("\033[0m")
        cf.write("\n[softmax]\n")
        print("node.attribute",node)
    elif(node.op_type == "Gemm"):
        print("\033[0;32;40m")
        print("[connected]")
        print("\033[0m")
        cf.write("\n[connected]\n")
        print("node.attribute",node)
        for input in graph.input:
            if(node.input[1] == input.name):
                print("input : ",input)
                print("\033[0;34;40m","output=%d"%input.type.tensor_type.shape.dim[0].dim_value,"\033[0m")
                cf.write("output=%d\n"%input.type.tensor_type.shape.dim[0].dim_value)

    else:
        print("\033[0;35;40m","Skip ",node.op_type,"\033[0m" )

cf.close()
wf.close()
#print(graph.output)

#print(onnx.helper.printable_graph(model.graph))

#onnx.checker.check_model(model)



print("data_type = ")
i = 0
for tgi in graph.node:
    print(tgi.name)
#(graph.initializer[i].raw_data):
#    print(graph.initializer[i])
#elif(graph.initializer[i].float_data):  #else int32_data, int64_data, double_data .etc  find at https://hexdocs.pm/onnxs/Onnx.TensorProto.html
#    print(graph.initializer[i].float_data)

print(conv_filter)



