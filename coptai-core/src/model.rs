use std::path::Path;

use candle_core::Device;

use crate::{
    error::CoptaiError,
    loaders::loader::{load_from_safetensors, ModelWeights},
    quantization::QuantizedTensorSerde,
    OptimizerConfig,
};

/// An unoptimised model loaded from disk.
pub struct CoptaiModel {
    pub weights: ModelWeights,
    pub source_paths: Vec<std::path::PathBuf>,
}

impl CoptaiModel {
    /// Convenience constructor — load all shards from a directory eagerly into RAM.
    /// Use this when you need F32 weights directly (inference, info, bench).
    /// For quantization passes, use `from_dir_lazy` to avoid a double load.
    pub fn from_dir(dir: &Path, device: &Device) -> Result<Self, CoptaiError> {
        let shard_paths = Self::collect_shards(dir)?;
        let refs: Vec<&Path> = shard_paths.iter().map(|p| p.as_path()).collect();
        let weights = load_from_safetensors(&refs, device)?;
        Ok(Self { weights, source_paths: shard_paths })
    }

    /// Discover shard paths without loading any weights into RAM.
    /// `weights` will be empty — the optimizer will stream-load during its pass.
    pub fn from_dir_lazy(dir: &Path) -> Result<Self, CoptaiError> {
        let shard_paths = Self::collect_shards(dir)?;
        tracing::info!(shards = shard_paths.len(), "Shards discovered (lazy — no data loaded yet)");
        Ok(Self {
            weights: std::collections::HashMap::new(),
            source_paths: shard_paths,
        })
    }

    fn collect_shards(dir: &Path) -> Result<Vec<std::path::PathBuf>, CoptaiError> {
        let mut shard_paths: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "safetensors")
                    .unwrap_or(false)
            })
            .collect();

        shard_paths.sort();

        if shard_paths.is_empty() {
            return Err(CoptaiError::Unsupported(format!(
                "No .safetensors files found in {}",
                dir.display()
            )));
        }

        Ok(shard_paths)
    }
}

/// A model that has passed through the optimisation pipeline.
pub struct OptimizedModel {
    pub inner: CoptaiModel,
    pub device: Device,
    pub config: OptimizerConfig,
    /// INT8-quantized weights produced by the quantization pass.
    pub quantized_weights: std::collections::HashMap<String, crate::quantization::int8::Int8QuantizedTensor>,
}

impl OptimizedModel {
    /// Return a reference to the underlying F32 weight map (non-quantized tensors).
    pub fn weights(&self) -> &ModelWeights {
        &self.inner.weights
    }

    /// Number of tensors in the original model (F32 + quantized combined).
    pub fn num_tensors(&self) -> usize {
        self.inner.weights.len() + self.quantized_weights.len()
    }

    /// Number of weight matrices that were INT8-quantized.
    pub fn num_quantized(&self) -> usize {
        self.quantized_weights.len()
    }

    /// Serialize the optimised model to `output_dir/model.safetensors`.
    ///
    /// Encoding convention:
    ///   `{name}.qint8`  — INT8 quantized bytes  (dtype I8,  shape [rows, cols])
    ///   `{name}.scales` — per-row scale factors  (dtype F32, shape [rows])
    ///   `{name}`        — non-quantized tensors  (kept at original dtype)
    pub fn save(&self, output_dir: &Path) -> Result<(), CoptaiError> {
        use safetensors::{serialize_to_file, tensor::TensorView, Dtype as SfDtype};
        use std::collections::HashMap;

        let out_path = output_dir.join("model.safetensors");

        // We need the raw bytes to stay alive for the duration of the call,
        // so we collect them into Vecs first, then build TensorViews that borrow them.
        struct OwnedTensor {
            dtype: SfDtype,
            shape: Vec<usize>,
            data: Vec<u8>,
        }

        let mut owned: Vec<(String, OwnedTensor)> = Vec::new();

        // --- Quantized tensors -------------------------------------------
        for (name, qt) in &self.quantized_weights {
            for (entry_name, dtype, shape, data) in qt.to_serde_entries(name) {
                owned.push((entry_name, OwnedTensor { dtype, shape, data }));
            }
        }

        // --- Non-quantized tensors ----------------------------------------
        for (name, tensor) in &self.inner.weights {
            let dtype = match tensor.dtype() {
                candle_core::DType::F32 => SfDtype::F32,
                candle_core::DType::F16 => SfDtype::F16,
                candle_core::DType::BF16 => SfDtype::BF16,
                candle_core::DType::F64 => SfDtype::F64,
                candle_core::DType::I64 => SfDtype::I64,
                candle_core::DType::U8 => SfDtype::U8,
                candle_core::DType::U32 => SfDtype::U32,
                // Weight files loaded via our loader are always one of the above dtypes.
                // Exotic candle-only dtypes (I16, I32, F8, …) cannot appear here.
                _ => unreachable!("unexpected dtype in saved weights: {:?}", tensor.dtype()),
            };
            let shape: Vec<usize> = tensor.dims().to_vec();
            let raw_bytes = tensor_to_bytes(tensor).map_err(CoptaiError::Candle)?;
            owned.push((
                name.clone(),
                OwnedTensor { dtype, shape, data: raw_bytes },
            ));
        }

        // Build TensorView map — borrows from `owned`.
        let views: HashMap<String, TensorView<'_>> = owned
            .iter()
            .map(|(name, ot)| {
                let view = TensorView::new(ot.dtype, ot.shape.clone(), &ot.data)
                    .map_err(|e| CoptaiError::Unsupported(e.to_string()))?;
                Ok::<_, CoptaiError>((name.clone(), view))
            })
            .collect::<Result<_, _>>()?;

        serialize_to_file(&views, &None, &out_path)
            .map_err(|e| CoptaiError::Unsupported(format!("safetensors write error: {e}")))?;

        tracing::info!(
            path = %out_path.display(),
            quantized = self.quantized_weights.len(),
            plain = self.inner.weights.len(),
            "Model saved"
        );
        Ok(())
    }
}

/// Extract the raw storage bytes from a candle Tensor without any copy of the
/// data beyond what candle already has in RAM.
fn tensor_to_bytes(tensor: &candle_core::Tensor) -> Result<Vec<u8>, candle_core::Error> {
    use candle_core::DType;
    // candle exposes `to_vec1::<u8>` only for U8 tensors.
    // For all other dtypes we go through the CPU flat storage and reinterpret.
    let flat = tensor.flatten_all()?;
    Ok(match flat.dtype() {
        DType::F32 => flat.to_vec1::<f32>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F16 => flat.to_vec1::<half::f16>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => flat.to_vec1::<half::bf16>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F64 => flat.to_vec1::<f64>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::I64 => flat.to_vec1::<i64>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::U8  => flat.to_vec1::<u8>()?,
        DType::U32 => flat.to_vec1::<u32>()?.iter().flat_map(|v| v.to_le_bytes()).collect(),
        // See dtype match in save(): exotic dtypes cannot appear in weight files we load.
        _ => unreachable!("unexpected dtype in tensor_to_bytes: {:?}", flat.dtype()),
    })
}
