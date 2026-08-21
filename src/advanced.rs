//! Opt-in backend, interchange, and generic-engine facilities.
//!
//! These items support custom algorithms and integrations that are not covered
//! by PiP's foundational or ready-model facades. Their invariants are catalogued
//! in the repository's `advanced_api.md`.

pub use crate::engines::observe::{MeanReducer, Reducer};
pub use crate::engines::soa::interaction::{
    Interaction, InteractionError, InteractionId, InteractionNodes, InteractionOrder,
    InteractionTopology, ObjId,
};
pub use crate::engines::soa::neighbor_list::{NeighborList, NeighborListError};
pub use crate::engines::soa::phys_obj::{AttrId, AttrsCore, AttrsMeta};
pub use crate::math::io::json::{
    FlatPayload, FromJsonPayload, JSON_SCHEMA_VERSION, SparsePayload, SparsePayloadParts,
    ToJsonPayload,
};
pub use crate::math::io::ndarray::NdarrayConvert;
pub use crate::math::io::string::TensorStringConvert;
pub use crate::math::scalar::ScalarSerde;
pub use crate::math::tensor::rank_2::matrix::{
    AntiSymmetricMatrix, DiagonalMatrix, LowerTriangularMatrix, MatrixBackend, RankNDense,
    RankNSparse, SparseMatrix, StrictLowerTriangularMatrix, StrictUpperTriangularMatrix,
    SymmetricMatrix, UpperTriangularMatrix,
};
pub use crate::math::tensor::rank_2::vector_list::{
    DynVectorList, HaarVectors, NNVectors, VectorListRand,
};
pub use crate::math::tensor::rank_n::dense_rand::{NUM_RNGS, TensorRandElement};
pub use crate::math::tensor::rank_n::tensor_trait::TensorTrait;
pub use crate::math::tensor::rank_n::{Backend, Dense, RowMajorLayout, Sparse, TensorResult};
pub use crate::sampling::{DynamicWeightedIndex, DynamicWeightedIndexError};
pub use crate::space::discrete::square_lattice::kernel::{
    Kernel, NearestNeighborKernel, PowerLawKernel, UniformDistanceKernel,
};
pub use crate::space::io::square_lattice::save_square_lattice;
pub use crate::space::space_trait::Space;

use crate::engines::soa::phys_obj::PhysObj;
use crate::math::Scalar;
use crate::space::SquareLattice;

/// Raw structure-of-arrays access for custom model and interchange code.
pub trait PhysObjAdvanced {
    /// Constructs a system state from raw metadata and attribute storage.
    fn from_raw_parts(metadata: AttrsMeta, attributes: AttrsCore) -> Self;

    /// Borrows the complete object metadata.
    fn metadata(&self) -> &AttrsMeta;

    /// Mutably borrows the complete object metadata.
    fn metadata_mut(&mut self) -> &mut AttrsMeta;

    /// Borrows the heterogeneous attribute store.
    fn attributes(&self) -> &AttrsCore;

    /// Mutably borrows the heterogeneous attribute store.
    fn attributes_mut(&mut self) -> &mut AttrsCore;
}

impl PhysObjAdvanced for PhysObj {
    fn from_raw_parts(metadata: AttrsMeta, attributes: AttrsCore) -> Self {
        PhysObj::new(metadata, attributes)
    }

    fn metadata(&self) -> &AttrsMeta {
        &self.meta
    }

    fn metadata_mut(&mut self) -> &mut AttrsMeta {
        &mut self.meta
    }

    fn attributes(&self) -> &AttrsCore {
        &self.core
    }

    fn attributes_mut(&mut self) -> &mut AttrsCore {
        &mut self.core
    }
}

/// Lower-level lattice transformations omitted from the foundational facade.
pub trait SquareLatticeAdvanced<T: Scalar> {
    /// Produces a nearest-site downsampled lattice with the requested shape.
    fn downsample(&self, target_shape: &[usize]) -> SquareLattice<T>;
}

impl<T: Scalar> SquareLatticeAdvanced<T> for SquareLattice<T> {
    fn downsample(&self, target_shape: &[usize]) -> SquareLattice<T> {
        SquareLattice::downsample(self, target_shape)
    }
}
