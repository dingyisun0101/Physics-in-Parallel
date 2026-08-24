//! Opt-in backend, interchange, and generic-engine API.

pub use crate::advanced::{
    AntiSymmetricMatrix, AttrId, AttrsCore, AttrsMeta, Backend, DEFAULT_RANDOM_MAX_THREADS, Dense,
    DiagonalMatrix, DynVectorList, DynamicWeightedIndex, DynamicWeightedIndexError, FlatPayload,
    FromJsonPayload, HaarVectors, Interaction, InteractionError, InteractionId, InteractionNodes,
    InteractionOrder, InteractionTopology, JSON_SCHEMA_VERSION, Kernel, LowerTriangularMatrix,
    MatrixBackend, MeanReducer, NNVectors, NUM_RNGS, NearestNeighborKernel, NeighborList,
    NeighborListError, ObjId, PhysObjAdvanced, PowerLawKernel, RankNDense, RankNSparse, Reducer,
    RowMajorLayout, ScalarSerde, Space, Sparse, SparseMatrix, SparsePayload, SparsePayloadParts,
    SquareLatticeAdvanced, StrictLowerTriangularMatrix, StrictUpperTriangularMatrix,
    SymmetricMatrix, TensorRandElement, TensorResult, TensorStringConvert, TensorTrait,
    ToJsonPayload, UniformDistanceKernel, UpperTriangularMatrix, VectorListRand,
    save_square_lattice,
};
