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

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use ahash::AHashMap;
use serde::Serialize;

use crate::math::io::json::{AttrsCorePayload, LabeledPayload, PhysObjPayload, ToJsonPayload};
use crate::math::{
    scalar::Scalar,
    tensor::rank_2::vector_list::{DynVectorList, VectorList},
};

pub type AttrId = usize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

    #[inline]
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
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
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
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
        Ok(col.get_vector(obj as isize))
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
        Ok(col.get_vector_mut(obj as isize))
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
        col.set_vector_from_slice(obj as isize, value);
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
}

impl ToJsonPayload for AttrsCore {
    type Payload = AttrsCorePayload;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        let mut labels: Vec<&str> = self.labels().collect();
        labels.sort_unstable();

        let mut attrs: Vec<LabeledPayload> = Vec::with_capacity(labels.len());
        for label in labels {
            let id = self.id_of(label).map_err(|e| {
                serde_json::Error::io(std::io::Error::other(format!(
                    "label disappeared during serialization: {e:?}"
                )))
            })?;
            let entry = self.entry(id).map_err(|e| {
                serde_json::Error::io(std::io::Error::other(format!(
                    "entry disappeared during serialization: {e:?}"
                )))
            })?;
            attrs.push(LabeledPayload {
                label: label.to_string(),
                payload: entry.data.serialize_value()?,
            });
        }

        Ok(AttrsCorePayload {
            n_objects: self.n_objects,
            num_attrs: self.len(),
            attrs,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PhysObj {
    pub meta: AttrsMeta,
    pub core: AttrsCore,
}

impl PhysObj {
    #[inline]
    pub fn new(meta: AttrsMeta, core: AttrsCore) -> Self {
        Self { meta, core }
    }

    #[inline]
    pub fn empty() -> Self {
        Self {
            meta: AttrsMeta::empty(),
            core: AttrsCore::empty(),
        }
    }

    #[inline]
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }

    pub fn save_to_json(&self, output_dir: &PathBuf, filename: &str) -> std::io::Result<()> {
        fs::create_dir_all(output_dir)?;
        let text = self
            .serialize()
            .map_err(|e| std::io::Error::other(format!("failed to serialize phys_obj: {e}")))?;

        let output_file = output_dir.join(filename);
        let mut file = File::create(output_file)?;
        file.write_all(text.as_bytes())?;
        Ok(())
    }
}

impl ToJsonPayload for PhysObj {
    type Payload = PhysObjPayload;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        Ok(PhysObjPayload {
            meta: serde_json::to_value(&self.meta)?,
            core: self.core.to_json_payload()?,
        })
    }
}
