//! Opt-in backend, interchange, and generic-engine facilities.
//!
//! These items support custom algorithms and integrations that are not covered
//! by PiP's foundational or ready-model facades. Their invariants are catalogued
//! in the repository's `advanced_api.md`.

pub use crate::engines::soa::interaction::{
    Interaction, InteractionError, InteractionId, InteractionNodes, InteractionOrder,
    InteractionTopology, ObjId,
};
pub use crate::engines::soa::neighbor_list::{NeighborList, NeighborListError};
pub use crate::engines::soa::phys_obj::{AttrId, AttrsCore, AttrsError, AttrsMeta};
pub use crate::math::tensor::rank_2::matrix::{
    AntiSymmetricMatrix, DiagonalMatrix, LowerTriangularMatrix, StrictLowerTriangularMatrix,
    StrictUpperTriangularMatrix, SymmetricMatrix, UpperTriangularMatrix,
};
pub use crate::sampling::{DynamicWeightedIndex, DynamicWeightedIndexError};
pub use crate::space::discrete::square_lattice::kernel::{
    Kernel, NearestNeighborKernel, PowerLawKernel, UniformDistanceKernel,
};

use crate::engines::soa::phys_obj::PhysObj;
use crate::math::{Backend, Matrix, Scalar, Tensor, VectorList};
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

/// Storage-native access for custom kernels.
///
/// Prefer each container's coordinate-based basic API. These flat-index and
/// backing-storage operations expose representation details and can have very
/// different costs for dense and sparse values.
pub trait RawStorage<T: Scalar> {
    fn value_at_index(&self, index: usize) -> Option<T>;
    fn set_value_at_index(&mut self, index: usize, value: T) -> bool;
    fn dense_values(&self) -> Option<&[T]>;
    fn dense_values_mut(&mut self) -> Option<&mut [T]>;
    fn stored_entries(&self) -> Vec<(usize, T)>;
    /// Visits stored entries without allocation in PiP containers. Dense storage
    /// includes zeros; sparse iteration order is unspecified. Custom implementors
    /// inherit a collecting fallback and may override it. Remains object-safe.
    fn for_each_stored_entry(&self, visitor: &mut dyn FnMut(usize, T)) {
        for (index, value) in self.stored_entries() {
            visitor(index, value);
        }
    }
}

impl<T: Scalar> RawStorage<T> for Tensor<T> {
    fn value_at_index(&self, index: usize) -> Option<T> {
        (index < self.size()).then(|| self.get_flat_unchecked(index))
    }

    fn set_value_at_index(&mut self, index: usize, value: T) -> bool {
        if index >= self.size() {
            return false;
        }
        self.set_flat_unchecked(index, value);
        true
    }

    fn dense_values(&self) -> Option<&[T]> {
        self.dense_values()
    }

    fn dense_values_mut(&mut self) -> Option<&mut [T]> {
        self.dense_values_mut()
    }

    fn for_each_stored_entry(&self, visitor: &mut dyn FnMut(usize, T)) {
        if let Some(entries) = self.sparse_entries() {
            for (index, value) in entries {
                visitor(index, value);
            }
        } else {
            for (index, &value) in self
                .dense_values()
                .expect("dense backend")
                .iter()
                .enumerate()
            {
                visitor(index, value);
            }
        }
    }

    fn stored_entries(&self) -> Vec<(usize, T)> {
        match self.backend() {
            Backend::Dense => self.values().enumerate().collect(),
            Backend::Sparse => self
                .sparse_entries()
                .expect("sparse representation has sparse entries")
                .collect(),
        }
    }
}

macro_rules! delegate_raw_storage {
    ($container:ident) => {
        impl<T: Scalar> RawStorage<T> for $container<T> {
            fn value_at_index(&self, index: usize) -> Option<T> {
                RawStorage::value_at_index(self.tensor(), index)
            }

            fn set_value_at_index(&mut self, index: usize, value: T) -> bool {
                RawStorage::set_value_at_index(self.tensor_mut(), index, value)
            }

            fn dense_values(&self) -> Option<&[T]> {
                RawStorage::dense_values(self.tensor())
            }

            fn dense_values_mut(&mut self) -> Option<&mut [T]> {
                RawStorage::dense_values_mut(self.tensor_mut())
            }

            fn for_each_stored_entry(&self, visitor: &mut dyn FnMut(usize, T)) {
                RawStorage::for_each_stored_entry(self.tensor(), visitor);
            }

            fn stored_entries(&self) -> Vec<(usize, T)> {
                RawStorage::stored_entries(self.tensor())
            }
        }
    };
}

delegate_raw_storage!(Matrix);
delegate_raw_storage!(VectorList);
