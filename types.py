from typing import Any, Callable, TypeVar, Union

from typeguard import TypeCheckError, check_type

# Base Types
Integer = int
Boolean = bool
IntegerPoint = tuple[int, int]
IntegerTuple = tuple[int, ...]
IntegerSet = frozenset[int]
Cell = tuple[int, IntegerPoint]
Grid = tuple[IntegerTuple, ...]
Indices = frozenset[IntegerPoint]
Object = frozenset[Cell]
ObjectCollection = frozenset[Object]

# Union Types
Numeric = Union[int, IntegerPoint, IntegerTuple, IntegerSet]
Element = Union[Grid, Object]
Patch = Union[Indices, Object]
PatchItem = Union[IntegerPoint, Cell]
Piece = Union[Grid, Patch]
Combinable = Union[
    IntegerPoint,
    IntegerTuple,
    IntegerSet,
    Cell,
    Grid,
    Indices,
    Object,
    ObjectCollection,
]
SetLike = Union[IntegerSet, Indices, Object, ObjectCollection]
SetLikeItem = Union[int, IntegerPoint, Cell, Object]
Comparable = Union[int, bool, Grid]
Container = Union[
    IntegerPoint,
    IntegerTuple,
    IntegerSet,
    Cell,
    Grid,
    Indices,
    Object,
    ObjectCollection,
]
ContainerItem = Union[int, bool, IntegerPoint, IntegerTuple, Cell, Indices, Object]
NestedContainer = Union[Grid, ObjectCollection, Indices]
NestedContainerItem = Union[IntegerTuple, Object]
SizedIterable = Union[
    IntegerPoint,
    IntegerTuple,
    IntegerSet,
    Cell,
    Grid,
    Indices,
    Object,
    ObjectCollection,
]
SizedIterableItem = Union[int, bool, IntegerPoint, IntegerTuple, Indices, Cell, Object]
OrderedIterable = Union[IntegerPoint, IntegerTuple, Indices, Cell, Grid]
OrderedIterableItem = Union[int, IntegerPoint, IntegerTuple]

# Universal Types
Anything = Union[
    int,
    bool,
    IntegerPoint,
    IntegerTuple,
    IntegerSet,
    Cell,
    Grid,
    Indices,
    Object,
    ObjectCollection,
]
AnythingTuple = tuple[Anything, ...]
NestedAnythingTuple = tuple[AnythingTuple, ...]
AnythingSet = frozenset[Anything]

# Function Types
Function = Callable[..., Union["Function", Anything]]

# Special Types
T = TypeVar("T")
ContainerItemTuple = tuple[ContainerItem, ...]
GenericContainer = Union[tuple[T, ...] | frozenset[T]]
FunctionContainer = GenericContainer[Function]
AnythingContainer = GenericContainer[Union[Anything, Function]]
AbsolutelyAnything = Union[
    Anything, AnythingTuple, AnythingSet, FunctionContainer, Function
]
AbsolutelyAnythingContainer = GenericContainer[AbsolutelyAnything]
NumericPoint = tuple[Numeric, Numeric]
AnythingSetLike = Union[SetLike, AbsolutelyAnything]


def assert_type(value: Any, type: Any) -> bool:
    try:
        check_type(value, type)
    except TypeCheckError:
        return False
    return True
