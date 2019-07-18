# @generated by generate_proto_mypy_stubs.py.  Do not edit!
from typing import (
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    cast,
)

from google.protobuf.message import (
    Message,
)

from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)

class Version(int):
    @classmethod
    def Name(cls, number: int) -> str: ...
    @classmethod
    def Value(cls, name: str) -> int: ...
    @classmethod
    def keys(cls) -> List[str]: ...
    @classmethod
    def values(cls) -> List[int]: ...
    @classmethod
    def items(cls) -> List[Tuple[str, int]]: ...
_START_VERSION = cast(Version, 0)
IR_VERSION_2017_10_10 = cast(Version, 1)
IR_VERSION_2017_10_30 = cast(Version, 2)
IR_VERSION = cast(Version, 3)

class AttributeProto(Message):
    class AttributeType(int):
        @classmethod
        def Name(cls, number: int) -> str: ...
        @classmethod
        def Value(cls, name: str) -> int: ...
        @classmethod
        def keys(cls) -> List[str]: ...
        @classmethod
        def values(cls) -> List[int]: ...
        @classmethod
        def items(cls) -> List[Tuple[str, int]]: ...
    UNDEFINED = cast(AttributeType, 0)
    FLOAT = cast(AttributeType, 1)
    INT = cast(AttributeType, 2)
    STRING = cast(AttributeType, 3)
    TENSOR = cast(AttributeType, 4)
    GRAPH = cast(AttributeType, 5)
    FLOATS = cast(AttributeType, 6)
    INTS = cast(AttributeType, 7)
    STRINGS = cast(AttributeType, 8)
    TENSORS = cast(AttributeType, 9)
    GRAPHS = cast(AttributeType, 10)
    
    name = ... # type: Text
    ref_attr_name = ... # type: Text
    doc_string = ... # type: Text
    type = ... # type: AttributeProto.AttributeType
    f = ... # type: float
    i = ... # type: int
    s = ... # type: bytes
    floats = ... # type: RepeatedScalarFieldContainer[float]
    ints = ... # type: RepeatedScalarFieldContainer[int]
    strings = ... # type: RepeatedScalarFieldContainer[bytes]
    
    @property
    def t(self) -> TensorProto: ...
    
    @property
    def g(self) -> GraphProto: ...
    
    @property
    def tensors(self) -> RepeatedCompositeFieldContainer[TensorProto]: ...
    
    @property
    def graphs(self) -> RepeatedCompositeFieldContainer[GraphProto]: ...
    
    def __init__(self,
        name : Optional[Text] = None,
        ref_attr_name : Optional[Text] = None,
        doc_string : Optional[Text] = None,
        type : Optional[AttributeProto.AttributeType] = None,
        f : Optional[float] = None,
        i : Optional[int] = None,
        s : Optional[bytes] = None,
        t : Optional[TensorProto] = None,
        g : Optional[GraphProto] = None,
        floats : Optional[Iterable[float]] = None,
        ints : Optional[Iterable[int]] = None,
        strings : Optional[Iterable[bytes]] = None,
        tensors : Optional[Iterable[TensorProto]] = None,
        graphs : Optional[Iterable[GraphProto]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> AttributeProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class ValueInfoProto(Message):
    name = ... # type: Text
    doc_string = ... # type: Text
    
    @property
    def type(self) -> TypeProto: ...
    
    def __init__(self,
        name : Optional[Text] = None,
        type : Optional[TypeProto] = None,
        doc_string : Optional[Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ValueInfoProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class NodeProto(Message):
    input = ... # type: RepeatedScalarFieldContainer[Text]
    output = ... # type: RepeatedScalarFieldContainer[Text]
    name = ... # type: Text
    op_type = ... # type: Text
    domain = ... # type: Text
    doc_string = ... # type: Text
    
    @property
    def attribute(self) -> RepeatedCompositeFieldContainer[AttributeProto]: ...
    
    def __init__(self,
        input : Optional[Iterable[Text]] = None,
        output : Optional[Iterable[Text]] = None,
        name : Optional[Text] = None,
        op_type : Optional[Text] = None,
        domain : Optional[Text] = None,
        attribute : Optional[Iterable[AttributeProto]] = None,
        doc_string : Optional[Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> NodeProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class ModelProto(Message):
    ir_version = ... # type: int
    producer_name = ... # type: Text
    producer_version = ... # type: Text
    domain = ... # type: Text
    model_version = ... # type: int
    doc_string = ... # type: Text
    
    @property
    def opset_import(self) -> RepeatedCompositeFieldContainer[OperatorSetIdProto]: ...
    
    @property
    def graph(self) -> GraphProto: ...
    
    @property
    def metadata_props(self) -> RepeatedCompositeFieldContainer[StringStringEntryProto]: ...
    
    def __init__(self,
        ir_version : Optional[int] = None,
        opset_import : Optional[Iterable[OperatorSetIdProto]] = None,
        producer_name : Optional[Text] = None,
        producer_version : Optional[Text] = None,
        domain : Optional[Text] = None,
        model_version : Optional[int] = None,
        doc_string : Optional[Text] = None,
        graph : Optional[GraphProto] = None,
        metadata_props : Optional[Iterable[StringStringEntryProto]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ModelProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class StringStringEntryProto(Message):
    key = ... # type: Text
    value = ... # type: Text
    
    def __init__(self,
        key : Optional[Text] = None,
        value : Optional[Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> StringStringEntryProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class GraphProto(Message):
    name = ... # type: Text
    doc_string = ... # type: Text
    
    @property
    def node(self) -> RepeatedCompositeFieldContainer[NodeProto]: ...
    
    @property
    def initializer(self) -> RepeatedCompositeFieldContainer[TensorProto]: ...
    
    @property
    def input(self) -> RepeatedCompositeFieldContainer[ValueInfoProto]: ...
    
    @property
    def output(self) -> RepeatedCompositeFieldContainer[ValueInfoProto]: ...
    
    @property
    def value_info(self) -> RepeatedCompositeFieldContainer[ValueInfoProto]: ...
    
    def __init__(self,
        node : Optional[Iterable[NodeProto]] = None,
        name : Optional[Text] = None,
        initializer : Optional[Iterable[TensorProto]] = None,
        doc_string : Optional[Text] = None,
        input : Optional[Iterable[ValueInfoProto]] = None,
        output : Optional[Iterable[ValueInfoProto]] = None,
        value_info : Optional[Iterable[ValueInfoProto]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> GraphProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class TensorProto(Message):
    class DataType(int):
        @classmethod
        def Name(cls, number: int) -> str: ...
        @classmethod
        def Value(cls, name: str) -> int: ...
        @classmethod
        def keys(cls) -> List[str]: ...
        @classmethod
        def values(cls) -> List[int]: ...
        @classmethod
        def items(cls) -> List[Tuple[str, int]]: ...
    UNDEFINED = cast(DataType, 0)
    FLOAT = cast(DataType, 1)
    UINT8 = cast(DataType, 2)
    INT8 = cast(DataType, 3)
    UINT16 = cast(DataType, 4)
    INT16 = cast(DataType, 5)
    INT32 = cast(DataType, 6)
    INT64 = cast(DataType, 7)
    STRING = cast(DataType, 8)
    BOOL = cast(DataType, 9)
    FLOAT16 = cast(DataType, 10)
    DOUBLE = cast(DataType, 11)
    UINT32 = cast(DataType, 12)
    UINT64 = cast(DataType, 13)
    COMPLEX64 = cast(DataType, 14)
    COMPLEX128 = cast(DataType, 15)
    BFLOAT16 = cast(DataType, 16)
    
    class DataLocation(int):
        @classmethod
        def Name(cls, number: int) -> str: ...
        @classmethod
        def Value(cls, name: str) -> int: ...
        @classmethod
        def keys(cls) -> List[str]: ...
        @classmethod
        def values(cls) -> List[int]: ...
        @classmethod
        def items(cls) -> List[Tuple[str, int]]: ...
    DEFAULT = cast(DataLocation, 0)
    EXTERNAL = cast(DataLocation, 1)
    
    class Segment(Message):
        begin = ... # type: int
        end = ... # type: int
        
        def __init__(self,
            begin : Optional[int] = None,
            end : Optional[int] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> TensorProto.Segment: ...
        def MergeFrom(self, other_msg: Message) -> None: ...
        def CopyFrom(self, other_msg: Message) -> None: ...
    
    dims = ... # type: RepeatedScalarFieldContainer[int]
    data_type = ... # type: int
    float_data = ... # type: RepeatedScalarFieldContainer[float]
    int32_data = ... # type: RepeatedScalarFieldContainer[int]
    string_data = ... # type: RepeatedScalarFieldContainer[bytes]
    int64_data = ... # type: RepeatedScalarFieldContainer[int]
    name = ... # type: Text
    doc_string = ... # type: Text
    raw_data = ... # type: bytes
    data_location = ... # type: TensorProto.DataLocation
    double_data = ... # type: RepeatedScalarFieldContainer[float]
    uint64_data = ... # type: RepeatedScalarFieldContainer[int]
    
    @property
    def segment(self) -> TensorProto.Segment: ...
    
    @property
    def external_data(self) -> RepeatedCompositeFieldContainer[StringStringEntryProto]: ...
    
    def __init__(self,
        dims : Optional[Iterable[int]] = None,
        data_type : Optional[int] = None,
        segment : Optional[TensorProto.Segment] = None,
        float_data : Optional[Iterable[float]] = None,
        int32_data : Optional[Iterable[int]] = None,
        string_data : Optional[Iterable[bytes]] = None,
        int64_data : Optional[Iterable[int]] = None,
        name : Optional[Text] = None,
        doc_string : Optional[Text] = None,
        raw_data : Optional[bytes] = None,
        external_data : Optional[Iterable[StringStringEntryProto]] = None,
        data_location : Optional[TensorProto.DataLocation] = None,
        double_data : Optional[Iterable[float]] = None,
        uint64_data : Optional[Iterable[int]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> TensorProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class TensorShapeProto(Message):
    class Dimension(Message):
        dim_value = ... # type: int
        dim_param = ... # type: Text
        denotation = ... # type: Text
        
        def __init__(self,
            dim_value : Optional[int] = None,
            dim_param : Optional[Text] = None,
            denotation : Optional[Text] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> TensorShapeProto.Dimension: ...
        def MergeFrom(self, other_msg: Message) -> None: ...
        def CopyFrom(self, other_msg: Message) -> None: ...
    
    
    @property
    def dim(self) -> RepeatedCompositeFieldContainer[TensorShapeProto.Dimension]: ...
    
    def __init__(self,
        dim : Optional[Iterable[TensorShapeProto.Dimension]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> TensorShapeProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class TypeProto(Message):
    class Tensor(Message):
        elem_type = ... # type: int
        
        @property
        def shape(self) -> TensorShapeProto: ...
        
        def __init__(self,
            elem_type : Optional[int] = None,
            shape : Optional[TensorShapeProto] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> TypeProto.Tensor: ...
        def MergeFrom(self, other_msg: Message) -> None: ...
        def CopyFrom(self, other_msg: Message) -> None: ...
    
    denotation = ... # type: Text
    
    @property
    def tensor_type(self) -> TypeProto.Tensor: ...
    
    def __init__(self,
        tensor_type : Optional[TypeProto.Tensor] = None,
        denotation : Optional[Text] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> TypeProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...

class OperatorSetIdProto(Message):
    domain = ... # type: Text
    version = ... # type: int
    
    def __init__(self,
        domain : Optional[Text] = None,
        version : Optional[int] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> OperatorSetIdProto: ...
    def MergeFrom(self, other_msg: Message) -> None: ...
    def CopyFrom(self, other_msg: Message) -> None: ...
