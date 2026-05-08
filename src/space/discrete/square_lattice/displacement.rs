/*!
Random source-target pair generation for square lattices.

Purpose:
    This module generates batches of lattice-coordinate pairs for Monte Carlo,
    stochastic dynamics, graph construction, and other algorithms that repeatedly
    choose a source site and a displacement. It is deliberately shape-aware but
    not boundary-aware.

Responsibility split:
    - This module guarantees that generated source coordinates are valid indices
      inside the square-lattice shape.
    - This module stores raw displacements and raw targets, where
      `target = source + displacement`.
    - `SquareLattice` decides how a raw coordinate is interpreted under
      `Periodic` or `Reflective` boundary conditions when callers later read or
      write lattice values.

Storage model:
    Sources, displacements, and targets are `VectorList<isize>` values with
    logical shape `[num_pairs, rank]`. Each row is one coordinate or vector.
*/

use rand::random_range;

use crate::math::prelude::{HaarVectors, NNVectors, TensorRandFiller, VectorList, VectorListRand};
use crate::space::discrete::square_lattice::kernel::{Kernel, KernelType, create_kernel};

#[derive(Clone)]
enum DispCache {
    Haar(HaarVectors),
    NN(NNVectors),
}

#[derive(Clone)]
pub enum SourceMode {
    Origin,
    RandomUniform,
    CustomFiller(TensorRandFiller),
}

#[derive(Clone)]
pub struct RandPairGenerator {
    shape: Vec<usize>,
    kernel: Box<dyn Kernel>,
    kernel_type: KernelType,
    source_mode: SourceMode,
    source_coords_cache: VectorList<isize>,
    displacement_cache: DispCache,
    displacement_coords_cache: VectorList<isize>,
    target_coords_cache: VectorList<isize>,
}

impl RandPairGenerator {
    pub fn new(
        shape: &[usize],
        kernel_type: KernelType,
        num_pairs: usize,
        source_mode: SourceMode,
        num_rngs: Option<usize>,
    ) -> Self {
        validate_shape(shape);
        assert!(
            num_pairs > 0,
            "RandPairGenerator::new: num_pairs must be > 0"
        );

        let rank = shape.len();
        validate_kernel_rank(kernel_type, rank);

        let kernel = create_kernel(kernel_type);
        let source_coords_cache = VectorList::empty(rank, num_pairs);
        let displacement_coords_cache = VectorList::empty(rank, num_pairs);
        let target_coords_cache = VectorList::empty(rank, num_pairs);
        let displacement_cache = match kernel_type {
            KernelType::NearestNeighbor { .. } => {
                DispCache::NN(NNVectors::new(rank, num_pairs, num_rngs))
            }
            KernelType::PowerLaw { .. } | KernelType::Uniform { .. } => {
                DispCache::Haar(HaarVectors::new(rank, num_pairs, num_rngs))
            }
        };

        Self {
            shape: shape.to_vec(),
            kernel,
            kernel_type,
            source_mode,
            source_coords_cache,
            displacement_cache,
            displacement_coords_cache,
            target_coords_cache,
        }
    }

    #[inline]
    pub fn refresh(&mut self) {
        self.refresh_sources();
        self.refresh_displacements();
        self.refresh_targets();
    }

    pub fn refresh_sources(&mut self) {
        match &mut self.source_mode {
            SourceMode::Origin => self.source_coords_cache.fill(0),
            SourceMode::RandomUniform => {
                let shape = self.shape.clone();
                self.source_coords_cache.par_for_each_vec_mut(|_, row| {
                    for (axis, coord) in row.iter_mut().enumerate() {
                        *coord = random_range(0..shape[axis]) as isize;
                    }
                });
            }
            SourceMode::CustomFiller(filler) => {
                filler.refresh(self.source_coords_cache.as_tensor_mut());
                normalize_sources_into_shape(&self.shape, &mut self.source_coords_cache);
            }
        }
    }

    pub fn refresh_displacements(&mut self) {
        let num_pairs = self.num_pairs();
        match (&self.kernel_type, &mut self.displacement_cache) {
            (KernelType::NearestNeighbor { .. }, DispCache::NN(nn)) => {
                nn.refresh();
                self.displacement_coords_cache = nn.vl.clone();
            }
            (KernelType::PowerLaw { .. } | KernelType::Uniform { .. }, DispCache::Haar(haar)) => {
                haar.refresh();
                let norms = self.kernel.sample(num_pairs);
                haar.vl.scale_vectors_by_list(&norms);
                self.displacement_coords_cache = haar.vl.cast_to::<isize>();
            }
            (kind, _) => panic!("kernel/cache mismatch in square-lattice pair generator: {kind:?}"),
        }
    }

    #[inline]
    pub fn refresh_targets(&mut self) {
        self.target_coords_cache = &self.source_coords_cache + &self.displacement_coords_cache;
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    #[inline]
    pub fn num_pairs(&self) -> usize {
        self.source_coords_cache.num_vectors()
    }

    #[inline]
    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    #[inline]
    pub fn sources(&self) -> &VectorList<isize> {
        &self.source_coords_cache
    }

    #[inline]
    pub fn displacements(&self) -> &VectorList<isize> {
        &self.displacement_coords_cache
    }

    #[inline]
    pub fn targets(&self) -> &VectorList<isize> {
        &self.target_coords_cache
    }

    #[inline]
    pub fn source(&self, i: isize) -> &[isize] {
        self.source_coords_cache.get_vec(i)
    }

    #[inline]
    pub fn displacement(&self, i: isize) -> &[isize] {
        self.displacement_coords_cache.get_vec(i)
    }

    #[inline]
    pub fn target(&self, i: isize) -> &[isize] {
        self.target_coords_cache.get_vec(i)
    }
}

fn validate_shape(shape: &[usize]) {
    assert!(
        !shape.is_empty(),
        "RandPairGenerator::new: shape must have at least one axis"
    );
    assert!(
        shape.iter().all(|&axis_len| axis_len > 0),
        "RandPairGenerator::new: shape must contain only nonzero axis lengths; got {shape:?}"
    );
}

fn validate_kernel_rank(kernel_type: KernelType, rank: usize) {
    if let KernelType::NearestNeighbor { d } = kernel_type {
        assert_eq!(
            d, rank,
            "NearestNeighbor kernel rank mismatch: kernel has d={d}, lattice shape has rank={rank}"
        );
    }
}

fn normalize_sources_into_shape(shape: &[usize], sources: &mut VectorList<isize>) {
    sources.par_for_each_vec_mut(|_, row| {
        for (axis, coord) in row.iter_mut().enumerate() {
            *coord = wrap_into_axis(*coord, shape[axis]);
        }
    });
}

#[inline]
fn wrap_into_axis(coord: isize, axis_len: usize) -> isize {
    let axis_len = axis_len as isize;
    let mut wrapped = coord % axis_len;
    if wrapped < 0 {
        wrapped += axis_len;
    }
    wrapped
}
