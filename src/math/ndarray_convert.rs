/*!
Compatibility re-export for ndarray conversion.

The canonical `NdarrayConvert` trait now lives under `crate::io::ndarray`.
This module is kept so existing math imports continue to work.
*/

pub use crate::io::ndarray::NdarrayConvert;
