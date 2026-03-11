/*!
Core attribute containers for SoA object storage.
*/

use ahash::AHashMap;
use serde::Serialize;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use crate::io::json::{
    AttrsCorePayload, LabeledPayload, PhysObjPayload, ToJsonPayload,
};
use crate::math::{
    scalar::Scalar,
    tensor::rank_2::vector_list::{DynVectorList, VectorList},
};

/// Stable attribute index.
pub type AttrId = usize;


// =============================================================================
// ------------------- AttrsMeta: The Metadata Wrapper -------------------------
// =============================================================================
/// Metadata for one attribute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AttrsMeta {
    pub id: AttrId,
    pub label: String,
    pub comment: String,
}

impl AttrsMeta {
    /// Empty/default metadata record.
    #[inline]
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    pub fn empty() -> Self {
        Self {
            id: 0,
            label: String::new(),
            comment: String::new(),
        }
    }

    #[inline]
    /// - Purpose: Serializes this metadata record into JSON text.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}




// =============================================================================
// --------------------- AttrsCore: Core Data Wrapper --------------------------
// =============================================================================
/// Errors for attribute metadata/core operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrsError {
    DuplicateLabel { label: String },
    UnknownLabel { label: String },
    InvalidVectorShape { dim: usize, n: usize },
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
    WrongVectorLen { expected: usize, got: usize },
}

/// Core typed vector-list storage keyed by attribute label.
#[derive(Debug, Clone, Default)]
pub struct AttrsCore {
    data: AHashMap<String, Box<dyn DynVectorList>>,
    n_objects: Option<usize>,
}

impl AttrsCore {
    #[inline]
    /// - Purpose: Constructs and returns a new instance.
    /// - Parameters:
    pub fn empty() -> Self {
        Self::default()
    }

    #[inline]
    /// - Purpose: Returns the current length/size.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    /// - Purpose: Checks whether `empty` condition is true.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    /// - Purpose: Checks whether a label exists in the attribute store.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup behavior.
    pub fn contains(&self, label: &str) -> bool {
        self.data.contains_key(label)
    }

    #[inline]
    /// - Purpose: Returns the object count invariant shared by all attributes when available.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn n_objects(&self) -> Option<usize> {
        self.n_objects
    }

    #[inline]
    /// - Purpose: Returns an iterator over all registered attribute labels.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.data.keys().map(|k| k.as_str())
    }

}

impl ToJsonPayload for AttrsCore {
    type Payload = AttrsCorePayload;

    fn to_json_payload(&self) -> Result<Self::Payload, serde_json::Error> {
        let mut labels: Vec<&str> = self.data.keys().map(|k| k.as_str()).collect();
        labels.sort_unstable();

        let mut attrs: Vec<LabeledPayload> = Vec::with_capacity(labels.len());
        for label in labels {
            let col = self
                .data
                .get(label)
                .ok_or_else(|| serde_json::Error::io(std::io::Error::other(format!(
                    "label disappeared during serialization: {label}"
                ))))?;
            attrs.push(LabeledPayload {
                label: label.to_string(),
                payload: col.serialize_value()?,
            });
        }

        Ok(AttrsCorePayload {
            n_objects: self.n_objects,
            num_attrs: self.data.len(),
            attrs,
        })
    }
}

impl AttrsCore {
    #[inline]
    /// - Purpose: Serializes this core attribute store into JSON text.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }
    /// - Purpose: Inserts a typed vector-list attribute under a unique label.
    /// - Parameters:
    ///   - `label` (`impl Into<String>`): Attribute label used as the storage key.
    ///   - `values` (`VectorList<T>`): Typed SoA vector-list payload stored for the label.
    pub fn insert<T: Scalar + Serialize + Copy + 'static>(
        &mut self,
        label: impl Into<String>,
        values: VectorList<T>,
    ) -> Result<(), AttrsError> {
        let label = label.into();
        if self.data.contains_key(&label) {
            return Err(AttrsError::DuplicateLabel { label });
        }

        let n = values.num_vectors();
        if let Some(expected) = self.n_objects {
            if n != expected {
                return Err(AttrsError::InconsistentObjectCount {
                    label,
                    expected,
                    got: n,
                });
            }
        }

        self.data.insert(label, Box::new(values));
        if self.n_objects.is_none() {
            self.n_objects = Some(n);
        }
        Ok(())
    }
    /// - Purpose: Allocates and inserts a new typed attribute column with shape metadata.
    /// - Parameters:
    ///   - `label` (`impl Into<String>`): Attribute label used as the storage key.
    ///   - `dim` (`usize`): Per-object vector dimension for the new attribute.
    ///   - `n` (`usize`): Number of objects/vectors to allocate for the new attribute.
    pub fn allocate<T: Scalar + Serialize + Copy + 'static>(
        &mut self,
        label: impl Into<String>,
        dim: usize,
        n: usize,
    ) -> Result<(), AttrsError> {
        if dim == 0 || n == 0 {
            return Err(AttrsError::InvalidVectorShape { dim, n });
        }
        self.insert(label, VectorList::<T>::empty(dim, n))
    }
    /// - Purpose: Removes an attribute by label and updates object-count tracking if store becomes empty.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key to remove from storage.
    pub fn remove(&mut self, label: &str) -> Result<(), AttrsError> {
        self.data
            .remove(label)
            .map(|_| {
                if self.data.is_empty() {
                    self.n_objects = None;
                }
            })
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: label.to_string(),
            })
    }
    /// - Purpose: Renames an existing attribute label while preserving its stored payload.
    /// - Parameters:
    ///   - `from` (`&str`): Existing attribute label key to rename.
    ///   - `to` (`&str`): Target attribute label key after rename.
    pub fn rename(&mut self, from: &str, to: &str) -> Result<(), AttrsError> {
        if from == to {
            return if self.data.contains_key(from) {
                Ok(())
            } else {
                Err(AttrsError::UnknownLabel {
                    label: from.to_string(),
                })
            };
        }

        if self.data.contains_key(to) {
            return Err(AttrsError::DuplicateLabel {
                label: to.to_string(),
            });
        }

        let col = self
            .data
            .remove(from)
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: from.to_string(),
            })?;
        self.data.insert(to.to_string(), col);
        Ok(())
    }
    /// - Purpose: Returns an immutable typed view of an attribute column.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup/downcast.
    pub fn get<T: Scalar + 'static>(&self, label: &str) -> Result<&VectorList<T>, AttrsError> {
        let col = self.data.get(label).ok_or_else(|| AttrsError::UnknownLabel {
            label: label.to_string(),
        })?;

        col.as_any()
            .downcast_ref::<VectorList<T>>()
            .ok_or_else(|| AttrsError::WrongType {
                label: label.to_string(),
                expected: std::any::type_name::<T>().to_string(),
                got: col.type_name().to_string(),
            })
    }
    /// - Purpose: Returns a mutable typed view of an attribute column.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup/downcast.
    pub fn get_mut<T: Scalar + 'static>(
        &mut self,
        label: &str,
    ) -> Result<&mut VectorList<T>, AttrsError> {
        let col = self
            .data
            .get_mut(label)
            .ok_or_else(|| AttrsError::UnknownLabel {
                label: label.to_string(),
            })?;
        let got = col.type_name().to_string();

        col.as_any_mut()
            .downcast_mut::<VectorList<T>>()
            .ok_or_else(|| AttrsError::WrongType {
                label: label.to_string(),
                expected: std::any::type_name::<T>().to_string(),
                got,
            })
    }
    /// - Purpose: Returns the vector value for one object from a typed attribute column.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup/downcast.
    ///   - `obj` (`usize`): Object row index to read from the attribute column.
    pub fn vector_of<T: Scalar + 'static>(
        &self,
        label: &str,
        obj: usize,
    ) -> Result<&[T], AttrsError>
    where
        T: Copy,
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
    /// - Purpose: Returns a mutable vector view for one object from a typed attribute column.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup/downcast.
    ///   - `obj` (`usize`): Object row index to mutate in the attribute column.
    pub fn vector_of_mut<T: Scalar + 'static>(
        &mut self,
        label: &str,
        obj: usize,
    ) -> Result<&mut [T], AttrsError>
    where
        T: Copy,
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
    /// - Purpose: Overwrites the vector value for one object in a typed attribute column.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup/downcast.
    ///   - `obj` (`usize`): Object row index to update in the attribute column.
    ///   - `value` (`&[T]`): Replacement vector data expected to match the attribute dimension.
    pub fn set_vector_of<T: Scalar + 'static>(
        &mut self,
        label: &str,
        obj: usize,
        value: &[T],
    ) -> Result<(), AttrsError> {
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
    /// - Purpose: Returns the per-object vector dimension for a registered attribute.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup.
    pub fn dim_of(&self, label: &str) -> Result<usize, AttrsError> {
        let col = self.data.get(label).ok_or_else(|| AttrsError::UnknownLabel {
            label: label.to_string(),
        })?;
        Ok(col.dim())
    }
    /// - Purpose: Returns the runtime scalar type name stored under an attribute label.
    /// - Parameters:
    ///   - `label` (`&str`): Attribute label key used for lookup.
    pub fn type_name_of(&self, label: &str) -> Result<&'static str, AttrsError> {
        let col = self.data.get(label).ok_or_else(|| AttrsError::UnknownLabel {
            label: label.to_string(),
        })?;
        Ok(col.type_name())
    }
}



// =============================================================================
// --------------------------------- PhysObj -----------------------------------
// =============================================================================
#[derive(Debug, Clone)]
pub struct PhysObj {
    pub meta: AttrsMeta,
    pub core: AttrsCore,
}

impl PhysObj {
    #[inline]
    /// - Purpose: Serializes this object container into pretty JSON text.
    /// - Parameters:
    ///   - (none): This function has no documented non-receiver parameters.
    pub fn serialize(&self) -> Result<String, serde_json::Error> {
        self.to_json_string()
    }

    #[inline]
    /// - Purpose: Saves this object container to a JSON file with both metadata and core attributes.
    /// - Parameters:
    ///   - `output_dir` (`&PathBuf`): Directory path where the JSON file will be created.
    ///   - `filename` (`&str`): Output JSON filename to create inside `output_dir`.
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
