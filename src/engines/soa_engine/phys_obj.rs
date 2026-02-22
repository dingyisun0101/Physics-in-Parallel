/*!
`PhysObj` is the core state container for particle/object-based simulations in
`engines::soa_engine`.

Design goals
------------
1. Keep **hot-path mechanics** fields explicit and contiguous.
2. Keep **non-core attributes** generic via a schema-driven column store.
3. Preserve **SoA memory layout** for cache-friendly axis-wise updates.

Storage model
-------------
- Core vector fields (`pos`, `vel`, `acc`) are stored as `VectorList<f64>` with shape `[D, n]`.
- Core scalar fields (`inv_mass`, `alive`, `kind`) are stored as `Vec<_>` of length `n`.
- Generic scalar attributes are stored as `Vec<Vec<f64>>`:
  - outer index = `AttrId`
  - inner index = `ObjId`
  - each column has length `n`

Schema model
------------
The schema maps attribute names to stable numeric IDs (`AttrId`), with a default value for
new objects and for initial allocation. This gives ergonomic name-based APIs while keeping
runtime access O(1) on numeric IDs.

Why this shape?
---------------
`HashMap<ObjId, HashMap<String, f64>>` is flexible but poor for simulation loops because it
destroys locality. A dict-of-columns (`name -> Vec<f64>`) keeps flexibility and remains fast.
*/

use ahash::AHashMap;

use crate::math::tensor::rank_2::vector_list::VectorList;

// ======================================================================================
// ---------------------------------- Type aliases --------------------------------------
// ======================================================================================

/// Stable object index in `PhysObj`.
pub type ObjId = usize;

/// Stable attribute index in the schema.
pub type AttrId = usize;

/// User-facing object kind tag.
pub type ObjKind = usize;

// ======================================================================================
// ----------------------------------- Error type ---------------------------------------
// ======================================================================================

/// Errors returned by `PhysObj` operations that depend on IDs or schema names.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysObjError {
    /// Object ID is out of range.
    InvalidObjId { obj: ObjId, n: usize },
    /// Attribute ID is out of range.
    InvalidAttrId { attr: AttrId, n_attrs: usize },
    /// Attribute name is not registered in the schema.
    UnknownAttrName { name: String },
    /// Attempted to register the same attribute name twice.
    DuplicateAttrName { name: String },
    /// Provided vector has invalid length for dimension `D`.
    WrongVectorLen { expected: usize, got: usize },
    /// Core scalar values must be finite and physically valid.
    InvalidCoreValue { field: &'static str, value: f64 },
}

// ======================================================================================
// ------------------------------ Attribute schema state --------------------------------
// ======================================================================================

/// Metadata for one attribute column.
#[derive(Debug, Clone)]
pub struct AttrMeta {
    /// Human-readable unique name (e.g., `"charge"`, `"radius"`, `"phase"`).
    pub name: String,
    /// Default value used when this attribute is created and when new objects are spawned.
    pub default: f64,
    /// Optional text annotation for downstream model code and docs.
    pub note: Option<String>,
}

/// Name/ID registry for generic per-object scalar attributes.
#[derive(Debug, Clone, Default)]
pub struct AttrSchema {
    metas: Vec<AttrMeta>,
    name_to_id: AHashMap<String, AttrId>,
}

impl AttrSchema {
    /// Create an empty schema.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered attributes.
    #[inline]
    pub fn len(&self) -> usize {
        self.metas.len()
    }

    /// True iff no attributes are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.metas.is_empty()
    }

    /// Immutable view of all metadata records.
    #[inline]
    pub fn metas(&self) -> &[AttrMeta] {
        &self.metas
    }

    /// Lookup attribute ID by name.
    #[inline]
    pub fn id_of(&self, name: &str) -> Option<AttrId> {
        self.name_to_id.get(name).copied()
    }

    /// Immutable metadata lookup by ID.
    #[inline]
    pub fn meta(&self, attr: AttrId) -> Option<&AttrMeta> {
        self.metas.get(attr)
    }

    /// Register a new attribute and return its ID.
    ///
    /// Invariant: names are unique and IDs are stable append-only indices.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        default: f64,
        note: Option<String>,
    ) -> Result<AttrId, PhysObjError> {
        if !default.is_finite() {
            return Err(PhysObjError::InvalidCoreValue {
                field: "attribute_default",
                value: default,
            });
        }

        let name = name.into();
        if self.name_to_id.contains_key(&name) {
            return Err(PhysObjError::DuplicateAttrName { name });
        }

        let id = self.metas.len();
        self.metas.push(AttrMeta {
            name: name.clone(),
            default,
            note,
        });
        self.name_to_id.insert(name, id);
        Ok(id)
    }
}

// ======================================================================================
// ------------------------------ PhysObj core container -------------------------------
// ======================================================================================

/**
Primary object-state container for continuous-object simulations.

Core fields
-----------
- `pos`, `vel`, `acc`: vector-valued fields of dimension `D`.
- `inv_mass`: scalar field used directly in force-to-acceleration updates.
- `alive`: soft lifecycle mask (allows deferred compaction).
- `kind`: lightweight type tag for downstream interaction rules.

Generic fields
--------------
- `attr_schema`: attribute registry (name/ID/default metadata).
- `attrs`: attribute columns (`attrs[attr_id][obj_id]`).

Invariants
----------
For every object field:
- length = `n`

For every vector field:
- shape = `[D, n]`

For generic attributes:
- `attrs.len() == attr_schema.len()`
- for all columns `c` in `attrs`: `c.len() == n`
*/
#[derive(Debug, Clone)]
pub struct PhysObj<const D: usize> {
    n: usize,

    // ------------------------------ core vector fields ------------------------------
    pos: VectorList<f64>,
    vel: VectorList<f64>,
    acc: VectorList<f64>,

    // ------------------------------ core scalar fields ------------------------------
    inv_mass: Vec<f64>,
    alive: Vec<bool>,
    kind: Vec<ObjKind>,

    // ------------------------------ generic attribute store -------------------------
    attr_schema: AttrSchema,
    attrs: Vec<Vec<f64>>, // attrs[attr_id][obj_id]
}

impl<const D: usize> PhysObj<D> {
    // ==========================================================================
    // ----------------------------- Constructors -------------------------------
    // ==========================================================================

    /**
    Construct a new container with `n` objects and default core values:
    - `pos`, `vel`, `acc` = 0
    - `inv_mass` = 1
    - `alive` = true
    - `kind` = 0
    - no generic attributes
    */
    pub fn empty(n: usize) -> Self {
        assert!(D > 0, "PhysObj::empty: D must be > 0");
        assert!(n > 0, "PhysObj::empty: n must be > 0");

        Self {
            n,
            pos: VectorList::<f64>::empty(D, n),
            vel: VectorList::<f64>::empty(D, n),
            acc: VectorList::<f64>::empty(D, n),
            inv_mass: vec![1.0; n],
            alive: vec![true; n],
            kind: vec![0; n],
            attr_schema: AttrSchema::new(),
            attrs: Vec::new(),
        }
    }

    /**
    Construct from explicit core state.

    Preconditions
    -------------
    - `inv_mass.len() == n`
    - `alive.len() == n`
    - `kind.len() == n`
    - all `inv_mass[i]` are finite and strictly positive
    - `pos/vel/acc` each have shape `[D, n]`
    */
    pub fn from_core(
        pos: VectorList<f64>,
        vel: VectorList<f64>,
        acc: VectorList<f64>,
        inv_mass: Vec<f64>,
        alive: Vec<bool>,
        kind: Vec<ObjKind>,
    ) -> Result<Self, PhysObjError> {
        assert!(D > 0, "PhysObj::from_core: D must be > 0");

        let n = inv_mass.len();
        assert!(n > 0, "PhysObj::from_core: n must be > 0");
        if alive.len() != n {
            return Err(PhysObjError::WrongVectorLen {
                expected: n,
                got: alive.len(),
            });
        }
        if kind.len() != n {
            return Err(PhysObjError::WrongVectorLen {
                expected: n,
                got: kind.len(),
            });
        }

        if pos.shape() != [D, n] {
            return Err(PhysObjError::WrongVectorLen {
                expected: D * n,
                got: pos.shape()[0] * pos.shape()[1],
            });
        }
        if vel.shape() != [D, n] {
            return Err(PhysObjError::WrongVectorLen {
                expected: D * n,
                got: vel.shape()[0] * vel.shape()[1],
            });
        }
        if acc.shape() != [D, n] {
            return Err(PhysObjError::WrongVectorLen {
                expected: D * n,
                got: acc.shape()[0] * acc.shape()[1],
            });
        }

        for &x in &inv_mass {
            if !x.is_finite() || x <= 0.0 {
                return Err(PhysObjError::InvalidCoreValue {
                    field: "inv_mass",
                    value: x,
                });
            }
        }

        Ok(Self {
            n,
            pos,
            vel,
            acc,
            inv_mass,
            alive,
            kind,
            attr_schema: AttrSchema::new(),
            attrs: Vec::new(),
        })
    }

    // ==========================================================================
    // ------------------------------- Meta / shape -----------------------------
    // ==========================================================================

    /// Compile-time spatial dimension `D`.
    #[inline]
    pub const fn dim() -> usize {
        D
    }

    /// Number of objects tracked by this container.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// True iff there are no objects.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Number of registered generic attributes.
    #[inline]
    pub fn num_attrs(&self) -> usize {
        self.attr_schema.len()
    }

    /// Borrow immutable attribute schema.
    #[inline]
    pub fn attr_schema(&self) -> &AttrSchema {
        &self.attr_schema
    }

    // ==========================================================================
    // ------------------------------ Core field accessors ----------------------
    // ==========================================================================

    /// Immutable position column view for object `obj` (length `D`).
    #[inline]
    pub fn pos_of(&self, obj: ObjId) -> Result<&[f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.pos.get_vector(obj as isize))
    }

    /// Mutable position column view for object `obj` (length `D`).
    #[inline]
    pub fn pos_of_mut(&mut self, obj: ObjId) -> Result<&mut [f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.pos.get_vector_mut(obj as isize))
    }

    /// Immutable velocity column view for object `obj` (length `D`).
    #[inline]
    pub fn vel_of(&self, obj: ObjId) -> Result<&[f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.vel.get_vector(obj as isize))
    }

    /// Mutable velocity column view for object `obj` (length `D`).
    #[inline]
    pub fn vel_of_mut(&mut self, obj: ObjId) -> Result<&mut [f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.vel.get_vector_mut(obj as isize))
    }

    /// Immutable acceleration column view for object `obj` (length `D`).
    #[inline]
    pub fn acc_of(&self, obj: ObjId) -> Result<&[f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.acc.get_vector(obj as isize))
    }

    /// Mutable acceleration column view for object `obj` (length `D`).
    #[inline]
    pub fn acc_of_mut(&mut self, obj: ObjId) -> Result<&mut [f64], PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.acc.get_vector_mut(obj as isize))
    }

    /// Set full position vector for object `obj`.
    pub fn set_pos(&mut self, obj: ObjId, value: &[f64]) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        if value.len() != D {
            return Err(PhysObjError::WrongVectorLen {
                expected: D,
                got: value.len(),
            });
        }
        self.pos.set_vector_from_slice(obj as isize, value);
        Ok(())
    }

    /// Set full velocity vector for object `obj`.
    pub fn set_vel(&mut self, obj: ObjId, value: &[f64]) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        if value.len() != D {
            return Err(PhysObjError::WrongVectorLen {
                expected: D,
                got: value.len(),
            });
        }
        self.vel.set_vector_from_slice(obj as isize, value);
        Ok(())
    }

    /// Set full acceleration vector for object `obj`.
    pub fn set_acc(&mut self, obj: ObjId, value: &[f64]) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        if value.len() != D {
            return Err(PhysObjError::WrongVectorLen {
                expected: D,
                got: value.len(),
            });
        }
        self.acc.set_vector_from_slice(obj as isize, value);
        Ok(())
    }

    /// Immutable inverse-mass scalar for object `obj`.
    #[inline]
    pub fn inv_mass_of(&self, obj: ObjId) -> Result<f64, PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.inv_mass[obj])
    }

    /// Update inverse mass for object `obj`. Must be finite and > 0.
    pub fn set_inv_mass(&mut self, obj: ObjId, value: f64) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(PhysObjError::InvalidCoreValue {
                field: "inv_mass",
                value,
            });
        }
        self.inv_mass[obj] = value;
        Ok(())
    }

    /// Read object liveness flag.
    #[inline]
    pub fn is_alive(&self, obj: ObjId) -> Result<bool, PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.alive[obj])
    }

    /// Set object liveness flag.
    #[inline]
    pub fn set_alive(&mut self, obj: ObjId, value: bool) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        self.alive[obj] = value;
        Ok(())
    }

    /// Read object kind tag.
    #[inline]
    pub fn kind_of(&self, obj: ObjId) -> Result<ObjKind, PhysObjError> {
        self.check_obj(obj)?;
        Ok(self.kind[obj])
    }

    /// Set object kind tag.
    #[inline]
    pub fn set_kind(&mut self, obj: ObjId, value: ObjKind) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        self.kind[obj] = value;
        Ok(())
    }

    /// Zero all acceleration vectors.
    #[inline]
    pub fn clear_acc(&mut self) {
        self.acc.fill(0.0);
    }

    // ==========================================================================
    // ------------------------------ Attribute schema APIs ---------------------
    // ==========================================================================

    /**
    Register a new scalar attribute.

    Side effect:
    - Appends one new column to `attrs` and fills it with the attribute default for all
      existing objects.
    */
    pub fn register_attr(
        &mut self,
        name: impl Into<String>,
        default: f64,
        note: Option<String>,
    ) -> Result<AttrId, PhysObjError> {
        let id = self.attr_schema.register(name, default, note)?;
        self.attrs.push(vec![default; self.n]);
        Ok(id)
    }

    /// Lookup attribute ID by name.
    #[inline]
    pub fn attr_id_of(&self, name: &str) -> Option<AttrId> {
        self.attr_schema.id_of(name)
    }

    /// Get attribute value by numeric IDs.
    pub fn attr(&self, obj: ObjId, attr: AttrId) -> Result<f64, PhysObjError> {
        self.check_obj(obj)?;
        self.check_attr(attr)?;
        Ok(self.attrs[attr][obj])
    }

    /// Set attribute value by numeric IDs.
    pub fn set_attr(&mut self, obj: ObjId, attr: AttrId, value: f64) -> Result<(), PhysObjError> {
        self.check_obj(obj)?;
        self.check_attr(attr)?;
        if !value.is_finite() {
            return Err(PhysObjError::InvalidCoreValue {
                field: "attribute_value",
                value,
            });
        }
        self.attrs[attr][obj] = value;
        Ok(())
    }

    /// Get attribute value by name.
    pub fn attr_by_name(&self, obj: ObjId, name: &str) -> Result<f64, PhysObjError> {
        let attr = self
            .attr_schema
            .id_of(name)
            .ok_or_else(|| PhysObjError::UnknownAttrName {
                name: name.to_string(),
            })?;
        self.attr(obj, attr)
    }

    /// Set attribute value by name.
    pub fn set_attr_by_name(
        &mut self,
        obj: ObjId,
        name: &str,
        value: f64,
    ) -> Result<(), PhysObjError> {
        let attr = self
            .attr_schema
            .id_of(name)
            .ok_or_else(|| PhysObjError::UnknownAttrName {
                name: name.to_string(),
            })?;
        self.set_attr(obj, attr, value)
    }

    // ==========================================================================
    // ------------------------------ Object lifecycle APIs ---------------------
    // ==========================================================================

    /**
    Append one object and return its new `ObjId`.

    New object defaults:
    - `pos/vel/acc = 0`
    - `inv_mass = 1`
    - `alive = true`
    - `kind = 0`
    - each generic attribute column = registered default
    */
    pub fn push_object(&mut self) -> ObjId {
        let obj = self.n;
        self.n += 1;

        // Extend core scalar columns.
        self.inv_mass.push(1.0);
        self.alive.push(true);
        self.kind.push(0);

        // Extend core vector columns by rebuilding from previous state.
        // VectorList currently models [D, n] as a fixed-shape matrix; append semantics are
        // implemented by resizing through explicit reconstruction.
        self.resize_vector_fields(self.n);

        // Extend generic attributes using per-attribute defaults from the schema.
        for (attr, col) in self.attrs.iter_mut().enumerate() {
            let default = self
                .attr_schema
                .meta(attr)
                .map(|m| m.default)
                .unwrap_or(0.0);
            col.push(default);
        }

        obj
    }

    /**
    Mark one object as dead (soft delete).

    This method does not remove rows immediately; it just toggles `alive = false`.
    Delayed compaction keeps IDs stable for interaction structures that still reference them.
    */
    pub fn kill_object(&mut self, obj: ObjId) -> Result<(), PhysObjError> {
        self.set_alive(obj, false)
    }

    // ==========================================================================
    // ------------------------------- Internal helpers -------------------------
    // ==========================================================================

    #[inline]
    fn check_obj(&self, obj: ObjId) -> Result<(), PhysObjError> {
        if obj >= self.n {
            return Err(PhysObjError::InvalidObjId { obj, n: self.n });
        }
        Ok(())
    }

    #[inline]
    fn check_attr(&self, attr: AttrId) -> Result<(), PhysObjError> {
        if attr >= self.attrs.len() {
            return Err(PhysObjError::InvalidAttrId {
                attr,
                n_attrs: self.attrs.len(),
            });
        }
        Ok(())
    }

    /// Rebuild all vector fields to shape `[D, new_n]`, preserving previous object values.
    fn resize_vector_fields(&mut self, new_n: usize) {
        let old_n = self.pos.num_vectors();

        let mut pos_new = VectorList::<f64>::empty(D, new_n);
        let mut vel_new = VectorList::<f64>::empty(D, new_n);
        let mut acc_new = VectorList::<f64>::empty(D, new_n);

        let copy_n = old_n.min(new_n);
        for i in 0..copy_n {
            let p = self.pos.get_vector(i as isize).to_vec();
            let v = self.vel.get_vector(i as isize).to_vec();
            let a = self.acc.get_vector(i as isize).to_vec();
            pos_new.set_vector_from_slice(i as isize, &p);
            vel_new.set_vector_from_slice(i as isize, &v);
            acc_new.set_vector_from_slice(i as isize, &a);
        }

        self.pos = pos_new;
        self.vel = vel_new;
        self.acc = acc_new;
    }
}
