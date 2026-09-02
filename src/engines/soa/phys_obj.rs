/*!
Structure-of-arrays object storage.

Purpose:
    This module stores many simulation objects as named attribute columns. Each
    attribute is a `VectorList<T>` with shape `[n_objects, dim]`, so one row is
    one object's vector-valued attribute. The container supports mixed scalar
    types by storing each column behind a runtime-typed `DynVectorList`.

Design:
    - `AttrsMeta` stores human-facing metadata for an object collection.
    - `AttrsCore` stores typed attribute columns keyed by label.
    - `PhysObj` bundles metadata and core attributes into one serializable
      simulation state container.

Invariants:
    - Every attribute column in one `AttrsCore` has the same `n_objects`.
    - Attribute labels are unique.
    - Attribute IDs are generated automatically and remain stable while an
      attribute exists.
    - Typed accessors check the requested scalar type at runtime and return a
      structured `AttrsError` on mismatch.
*/

use std::fmt;

use ahash::AHashMap;
use num_complex::Complex;
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;

use crate::math::io::json::JSON_SCHEMA_VERSION;
use crate::math::{
    scalar::Scalar,
    tensor::rank_2::vector_list::{DynVectorList, VectorList},
};

pub type AttrId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttrsMeta {
    pub id: AttrId,
    pub label: String,
    pub comment: String,
}

impl AttrsMeta {
    #[inline]
    pub fn empty() -> Self {
        Self {
            id: 0,
            label: String::new(),
            comment: String::new(),
        }
    }

    #[inline]
    pub fn new(id: AttrId, label: impl Into<String>, comment: impl Into<String>) -> Self {
        Self {
            id,
            label: label.into(),
            comment: comment.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrsError {
    DuplicateLabel {
        label: String,
    },
    UnknownLabel {
        label: String,
    },
    UnknownId {
        id: AttrId,
    },
    InvalidVectorShape {
        dim: usize,
        n: usize,
    },
    InconsistentObjectCount {
        label: String,
        expected: usize,
        got: usize,
    },
    ObjOutOfBounds {
        label: String,
        obj: usize,
        n: usize,
    },
    WrongType {
        label: String,
        expected: String,
        got: String,
    },
    WrongVectorLen {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for AttrsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLabel { label } => write!(f, "attribute label `{label}` already exists"),
            Self::UnknownLabel { label } => write!(f, "unknown attribute label `{label}`"),
            Self::UnknownId { id } => write!(f, "unknown attribute id {id}"),
            Self::InvalidVectorShape { dim, n } => write!(
                f,
                "attribute vector shape requires positive dimension and object count; got dim={dim}, n={n}"
            ),
            Self::InconsistentObjectCount {
                label,
                expected,
                got,
            } => write!(
                f,
                "attribute `{label}` has {got} objects; expected {expected}"
            ),
            Self::ObjOutOfBounds { label, obj, n } => write!(
                f,
                "object index {obj} is out of bounds for attribute `{label}` with {n} objects"
            ),
            Self::WrongType {
                label,
                expected,
                got,
            } => write!(
                f,
                "attribute `{label}` has scalar type `{got}`; expected `{expected}`"
            ),
            Self::WrongVectorLen { expected, got } => {
                write!(f, "attribute vector has length {got}; expected {expected}")
            }
        }
    }
}

impl std::error::Error for AttrsError {}

#[derive(Debug, Clone)]
struct AttrEntry {
    label: String,
    data: Box<dyn DynVectorList>,
}

#[derive(Debug, Clone, Default)]
pub struct AttrsCore {
    label_to_id: AHashMap<String, AttrId>,
    entries: Vec<Option<AttrEntry>>,
    n_objects: Option<usize>,
}

impl AttrsCore {
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.label_to_id.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.label_to_id.is_empty()
    }

    #[inline]
    pub fn contains(&self, label: &str) -> bool {
        self.label_to_id.contains_key(label)
    }

    #[inline]
    pub fn n_objects(&self) -> Option<usize> {
        self.n_objects
    }

    #[inline]
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.entries
            .iter()
            .filter_map(|entry| entry.as_ref().map(|entry| entry.label.as_str()))
    }

    #[inline]
    pub fn id_of(&self, label: &str) -> Result<AttrId, AttrsError> {
        self.label_to_id
            .get(label)
            .copied()
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: label.to_string(),
            })
    }

    #[inline]
    pub fn label_of(&self, id: AttrId) -> Result<&str, AttrsError> {
        Ok(&self.entry(id)?.label)
    }

    pub fn insert<T>(
        &mut self,
        label: impl Into<String>,
        values: VectorList<T>,
    ) -> Result<(), AttrsError>
    where
        T: Scalar + Serialize + Copy + 'static,
    {
        let label = label.into();
        if self.label_to_id.contains_key(&label) {
            return Err(AttrsError::DuplicateLabel { label });
        }

        let n = values.num_vectors();
        if let Some(expected) = self.n_objects
            && n != expected
        {
            return Err(AttrsError::InconsistentObjectCount {
                label,
                expected,
                got: n,
            });
        }

        let id = self.entries.len();
        self.entries.push(Some(AttrEntry {
            label: label.clone(),
            data: Box::new(values),
        }));
        self.label_to_id.insert(label, id);
        if self.n_objects.is_none() {
            self.n_objects = Some(n);
        }
        Ok(())
    }

    pub fn allocate<T>(
        &mut self,
        label: impl Into<String>,
        dim: usize,
        n: usize,
    ) -> Result<(), AttrsError>
    where
        T: Scalar + Serialize + Copy + 'static,
    {
        if dim == 0 || n == 0 {
            return Err(AttrsError::InvalidVectorShape { dim, n });
        }
        self.insert(label, VectorList::<T>::empty(dim, n))
    }

    pub fn remove(&mut self, label: &str) -> Result<(), AttrsError> {
        let id = self
            .label_to_id
            .remove(label)
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: label.to_string(),
            })?;
        self.entries[id] = None;
        if self.label_to_id.is_empty() {
            self.n_objects = None;
        }
        Ok(())
    }

    pub fn rename(&mut self, from: &str, to: &str) -> Result<(), AttrsError> {
        if from == to {
            return if self.label_to_id.contains_key(from) {
                Ok(())
            } else {
                Err(AttrsError::UnknownLabel {
                    label: from.to_string(),
                })
            };
        }

        if self.label_to_id.contains_key(to) {
            return Err(AttrsError::DuplicateLabel {
                label: to.to_string(),
            });
        }

        let id = self
            .label_to_id
            .remove(from)
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: from.to_string(),
            })?;
        let entry = self.entry_mut(id)?;
        entry.label = to.to_string();
        self.label_to_id.insert(to.to_string(), id);
        Ok(())
    }

    pub fn get<T: Scalar + 'static>(&self, label: &str) -> Result<&VectorList<T>, AttrsError> {
        let id = self.id_of(label)?;
        self.get_by_id(id)
    }

    pub fn get_by_id<T: Scalar + 'static>(&self, id: AttrId) -> Result<&VectorList<T>, AttrsError> {
        let entry = self.entry(id)?;
        entry
            .data
            .as_any()
            .downcast_ref::<VectorList<T>>()
            .ok_or_else(|| AttrsError::WrongType {
                label: entry.label.clone(),
                expected: std::any::type_name::<T>().to_string(),
                got: entry.data.type_name().to_string(),
            })
    }

    pub fn get_mut<T: Scalar + 'static>(
        &mut self,
        label: &str,
    ) -> Result<&mut VectorList<T>, AttrsError> {
        let id = self.id_of(label)?;
        self.get_by_id_mut(id)
    }

    pub fn get_by_id_mut<T: Scalar + 'static>(
        &mut self,
        id: AttrId,
    ) -> Result<&mut VectorList<T>, AttrsError> {
        let entry = self.entry_mut(id)?;
        Self::entry_data_mut(entry)
    }

    pub fn get_two_mut<T: Scalar + 'static>(
        &mut self,
        first: &str,
        second: &str,
    ) -> Result<(&mut VectorList<T>, &mut VectorList<T>), AttrsError> {
        let first_id = self.id_of(first)?;
        let second_id = self.id_of(second)?;
        if first_id == second_id {
            return Err(AttrsError::DuplicateLabel {
                label: first.to_string(),
            });
        }

        let [first_entry, second_entry] = self
            .entries
            .get_disjoint_mut([first_id, second_id])
            .expect("distinct valid attribute ids should be disjoint");

        let first_entry = first_entry
            .as_mut()
            .ok_or(AttrsError::UnknownId { id: first_id })?;
        let second_entry = second_entry
            .as_mut()
            .ok_or(AttrsError::UnknownId { id: second_id })?;

        Ok((
            Self::entry_data_mut(first_entry)?,
            Self::entry_data_mut(second_entry)?,
        ))
    }

    #[allow(clippy::type_complexity)]
    pub fn get_three_mut<T: Scalar + 'static>(
        &mut self,
        first: &str,
        second: &str,
        third: &str,
    ) -> Result<(&mut VectorList<T>, &mut VectorList<T>, &mut VectorList<T>), AttrsError> {
        let first_id = self.id_of(first)?;
        let second_id = self.id_of(second)?;
        let third_id = self.id_of(third)?;
        if first_id == second_id || first_id == third_id {
            return Err(AttrsError::DuplicateLabel {
                label: first.to_string(),
            });
        }
        if second_id == third_id {
            return Err(AttrsError::DuplicateLabel {
                label: second.to_string(),
            });
        }

        let [first_entry, second_entry, third_entry] = self
            .entries
            .get_disjoint_mut([first_id, second_id, third_id])
            .expect("distinct valid attribute ids should be disjoint");

        let first_entry = first_entry
            .as_mut()
            .ok_or(AttrsError::UnknownId { id: first_id })?;
        let second_entry = second_entry
            .as_mut()
            .ok_or(AttrsError::UnknownId { id: second_id })?;
        let third_entry = third_entry
            .as_mut()
            .ok_or(AttrsError::UnknownId { id: third_id })?;

        Ok((
            Self::entry_data_mut(first_entry)?,
            Self::entry_data_mut(second_entry)?,
            Self::entry_data_mut(third_entry)?,
        ))
    }

    fn entry_data_mut<T: Scalar + 'static>(
        entry: &mut AttrEntry,
    ) -> Result<&mut VectorList<T>, AttrsError> {
        let got = entry.data.type_name().to_string();

        entry
            .data
            .as_any_mut()
            .downcast_mut::<VectorList<T>>()
            .ok_or_else(|| AttrsError::WrongType {
                label: entry.label.clone(),
                expected: std::any::type_name::<T>().to_string(),
                got,
            })
    }

    pub fn vector_of<T>(&self, label: &str, obj: usize) -> Result<&[T], AttrsError>
    where
        T: Scalar + Copy + 'static,
    {
        let col = self.get::<T>(label)?;
        let n = col.num_vectors();
        if obj >= n {
            return Err(AttrsError::ObjOutOfBounds {
                label: label.to_string(),
                obj,
                n,
            });
        }
        Ok(col.vector(obj as isize))
    }

    pub fn vector_of_mut<T>(&mut self, label: &str, obj: usize) -> Result<&mut [T], AttrsError>
    where
        T: Scalar + Copy + 'static,
    {
        let col = self.get_mut::<T>(label)?;
        let n = col.num_vectors();
        if obj >= n {
            return Err(AttrsError::ObjOutOfBounds {
                label: label.to_string(),
                obj,
                n,
            });
        }
        Ok(col.vector_mut(obj as isize))
    }

    pub fn set_vector_of<T>(
        &mut self,
        label: &str,
        obj: usize,
        value: &[T],
    ) -> Result<(), AttrsError>
    where
        T: Scalar + Copy + 'static,
    {
        let col = self.get_mut::<T>(label)?;
        let n = col.num_vectors();
        if obj >= n {
            return Err(AttrsError::ObjOutOfBounds {
                label: label.to_string(),
                obj,
                n,
            });
        }
        if value.len() != col.dim() {
            return Err(AttrsError::WrongVectorLen {
                expected: col.dim(),
                got: value.len(),
            });
        }
        col.set_vector(obj as isize, value);
        Ok(())
    }

    pub fn dim_of(&self, label: &str) -> Result<usize, AttrsError> {
        let id = self.id_of(label)?;
        self.dim_of_id(id)
    }

    #[inline]
    pub fn dim_of_id(&self, id: AttrId) -> Result<usize, AttrsError> {
        Ok(self.entry(id)?.data.dim())
    }

    pub fn type_name_of(&self, label: &str) -> Result<&'static str, AttrsError> {
        let id = self.id_of(label)?;
        self.type_name_of_id(id)
    }

    #[inline]
    pub fn type_name_of_id(&self, id: AttrId) -> Result<&'static str, AttrsError> {
        Ok(self.entry(id)?.data.type_name())
    }

    fn entry(&self, id: AttrId) -> Result<&AttrEntry, AttrsError> {
        self.entries
            .get(id)
            .and_then(|entry| entry.as_ref())
            .ok_or(AttrsError::UnknownId { id })
    }

    fn entry_mut(&mut self, id: AttrId) -> Result<&mut AttrEntry, AttrsError> {
        self.entries
            .get_mut(id)
            .and_then(|entry| entry.as_mut())
            .ok_or(AttrsError::UnknownId { id })
    }

    /// Inserts one decoded attribute at its persisted stable slot.
    fn insert_decoded<T>(
        &mut self,
        id: AttrId,
        label: String,
        values: VectorList<T>,
    ) -> Result<(), String>
    where
        T: Scalar + Serialize + Copy + 'static,
    {
        if id >= self.entries.len() {
            return Err(format!(
                "attribute id {id} exceeds persisted slot count {}",
                self.entries.len()
            ));
        }
        if self.entries[id].is_some() {
            return Err(format!("duplicate attribute id {id}"));
        }
        if self.label_to_id.contains_key(&label) {
            return Err(format!("duplicate attribute label '{label}'"));
        }
        let count = values.num_vectors();
        if let Some(expected) = self.n_objects
            && count != expected
        {
            return Err(format!(
                "attribute '{label}' object count mismatch: expected {expected}, got {count}"
            ));
        }
        self.entries[id] = Some(AttrEntry {
            label: label.clone(),
            data: Box::new(values),
        });
        self.label_to_id.insert(label, id);
        if self.n_objects.is_none() {
            self.n_objects = Some(count);
        }
        Ok(())
    }
}

impl Serialize for AttrsCore {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut document = serializer.serialize_struct("AttrsCore", 6)?;
        document.serialize_field("kind", "attrs_core")?;
        document.serialize_field("version", &JSON_SCHEMA_VERSION)?;
        document.serialize_field("n_objects", &self.n_objects)?;
        document.serialize_field("num_attrs", &self.len())?;
        document.serialize_field("slot_count", &self.entries.len())?;
        document.serialize_field("attrs", &AttributeSequence { core: self })?;
        document.end()
    }
}

/// Streams active attributes in stable slot order without materializing JSON values.
struct AttributeSequence<'a> {
    core: &'a AttrsCore,
}

impl Serialize for AttributeSequence<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.core.len()))?;
        for (id, entry) in self.core.entries.iter().enumerate() {
            if let Some(entry) = entry {
                sequence.serialize_element(&AttributeRef { id, entry })?;
            }
        }
        sequence.end()
    }
}

/// Borrowed representation of one heterogeneous typed attribute.
struct AttributeRef<'a> {
    id: AttrId,
    entry: &'a AttrEntry,
}

impl Serialize for AttributeRef<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut attribute = serializer.serialize_struct("Attribute", 4)?;
        attribute.serialize_field("id", &self.id)?;
        attribute.serialize_field("label", &self.entry.label)?;
        attribute.serialize_field("scalar", self.entry.data.scalar_kind())?;
        attribute.serialize_field("payload", self.entry.data.as_ref())?;
        attribute.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AttrsCoreDocument {
    kind: String,
    version: u32,
    n_objects: Option<usize>,
    num_attrs: usize,
    slot_count: usize,
    attrs: Vec<EncodedAttribute>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EncodedAttribute {
    id: AttrId,
    label: String,
    scalar: String,
    payload: Box<RawValue>,
}

impl<'de> Deserialize<'de> for AttrsCore {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = AttrsCoreDocument::deserialize(deserializer)?;
        if document.kind != "attrs_core" {
            return Err(serde::de::Error::custom(format!(
                "AttrsCore kind must be 'attrs_core', got '{}'",
                document.kind
            )));
        }
        if document.version != JSON_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format!(
                "AttrsCore version mismatch: expected {JSON_SCHEMA_VERSION}, got {}",
                document.version
            )));
        }
        if document.num_attrs != document.attrs.len() {
            return Err(serde::de::Error::custom(format!(
                "AttrsCore attribute count mismatch: declared {}, got {}",
                document.num_attrs,
                document.attrs.len()
            )));
        }
        if document.num_attrs > document.slot_count {
            return Err(serde::de::Error::custom(
                "AttrsCore active attribute count exceeds slot count",
            ));
        }

        let mut entries = Vec::new();
        entries
            .try_reserve_exact(document.slot_count)
            .map_err(serde::de::Error::custom)?;
        entries.resize_with(document.slot_count, || None);
        let mut core = AttrsCore {
            label_to_id: AHashMap::with_capacity(document.num_attrs),
            entries,
            n_objects: None,
        };
        for attribute in document.attrs {
            decode_attribute(&mut core, attribute).map_err(serde::de::Error::custom)?;
        }
        if core.n_objects != document.n_objects {
            return Err(serde::de::Error::custom(format!(
                "AttrsCore object count mismatch: declared {:?}, decoded {:?}",
                document.n_objects, core.n_objects
            )));
        }
        Ok(core)
    }
}

fn decode_attribute(core: &mut AttrsCore, attribute: EncodedAttribute) -> Result<(), String> {
    macro_rules! decode {
        ($scalar:ty) => {{
            let values = serde_json::from_str::<VectorList<$scalar>>(attribute.payload.get())
                .map_err(|error| {
                    format!(
                        "failed to decode attribute '{}' as {}: {error}",
                        attribute.label, attribute.scalar
                    )
                })?;
            core.insert_decoded(attribute.id, attribute.label, values)
        }};
    }
    match attribute.scalar.as_str() {
        "f32" => decode!(f32),
        "f64" => decode!(f64),
        "i8" => decode!(i8),
        "i16" => decode!(i16),
        "i32" => decode!(i32),
        "i64" => decode!(i64),
        "i128" => decode!(i128),
        "isize" => decode!(isize),
        "u8" => decode!(u8),
        "u16" => decode!(u16),
        "u32" => decode!(u32),
        "u64" => decode!(u64),
        "u128" => decode!(u128),
        "usize" => decode!(usize),
        "complex_f32" => decode!(Complex<f32>),
        "complex_f64" => decode!(Complex<f64>),
        scalar => Err(format!(
            "attribute '{}' uses unsupported scalar kind '{scalar}'",
            attribute.label
        )),
    }
}

#[derive(Debug, Clone)]
pub struct PhysObj {
    /// Advanced generic metadata storage. Model users normally call the
    /// descriptive accessors on `PhysObj`.
    pub(crate) meta: AttrsMeta,
    /// Advanced heterogeneous attribute storage. Model users normally call
    /// the typed attribute accessors on `PhysObj`.
    pub(crate) core: AttrsCore,
}

impl PhysObj {
    #[inline]
    pub(crate) fn new(meta: AttrsMeta, core: AttrsCore) -> Self {
        Self { meta, core }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            meta: AttrsMeta::empty(),
            core: AttrsCore::empty(),
        }
    }

    /// Returns the model object's stable metadata identifier.
    #[inline]
    pub const fn id(&self) -> AttrId {
        self.meta.id
    }

    /// Returns the model object's human-facing label.
    #[inline]
    pub fn label(&self) -> &str {
        &self.meta.label
    }

    /// Returns the model object's optional descriptive comment.
    #[inline]
    pub fn comment(&self) -> &str {
        &self.meta.comment
    }

    /// Returns the common object count when attributes have been allocated.
    #[inline]
    pub fn num_objects(&self) -> Option<usize> {
        self.core.n_objects()
    }

    /// Reports whether one named attribute exists.
    #[inline]
    pub fn has_attribute(&self, label: &str) -> bool {
        self.core.contains(label)
    }

    /// Borrows one typed attribute column by canonical label.
    #[inline]
    pub fn attribute<T: Scalar + 'static>(
        &self,
        label: &str,
    ) -> Result<&VectorList<T>, AttrsError> {
        self.core.get(label)
    }

    /// Mutably borrows one typed attribute column by canonical label.
    #[inline]
    pub fn attribute_mut<T: Scalar + 'static>(
        &mut self,
        label: &str,
    ) -> Result<&mut VectorList<T>, AttrsError> {
        self.core.get_mut(label)
    }

    /// Borrows one object's vector from a named typed attribute.
    #[inline]
    pub fn attribute_vector<T: Scalar + 'static>(
        &self,
        label: &str,
        object: usize,
    ) -> Result<&[T], AttrsError> {
        self.core.vector_of(label, object)
    }

    /// Mutably borrows one object's vector from a named typed attribute.
    #[inline]
    pub fn attribute_vector_mut<T: Scalar + 'static>(
        &mut self,
        label: &str,
        object: usize,
    ) -> Result<&mut [T], AttrsError> {
        self.core.vector_of_mut(label, object)
    }

    /// Replaces one object's vector in a named typed attribute.
    #[inline]
    pub fn set_attribute_vector<T: Scalar + 'static>(
        &mut self,
        label: &str,
        object: usize,
        values: &[T],
    ) -> Result<(), AttrsError> {
        self.core.set_vector_of(label, object, values)
    }
}

impl Serialize for PhysObj {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut document = serializer.serialize_struct("PhysObj", 4)?;
        document.serialize_field("kind", "phys_obj")?;
        document.serialize_field("version", &JSON_SCHEMA_VERSION)?;
        document.serialize_field("meta", &self.meta)?;
        document.serialize_field("core", &self.core)?;
        document.end()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysObjDocument {
    kind: String,
    version: u32,
    meta: AttrsMeta,
    core: AttrsCore,
}

impl<'de> Deserialize<'de> for PhysObj {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let document = PhysObjDocument::deserialize(deserializer)?;
        if document.kind != "phys_obj" {
            return Err(serde::de::Error::custom(format!(
                "PhysObj kind must be 'phys_obj', got '{}'",
                document.kind
            )));
        }
        if document.version != JSON_SCHEMA_VERSION {
            return Err(serde::de::Error::custom(format!(
                "PhysObj version mismatch: expected {JSON_SCHEMA_VERSION}, got {}",
                document.version
            )));
        }
        Ok(Self {
            meta: document.meta,
            core: document.core,
        })
    }
}
