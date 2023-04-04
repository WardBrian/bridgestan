use crate::ffi;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::str::Utf8Error;

// This is more or less equivalent to manually defining Display and From<other error types>
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum BridgeStanError {
    #[error("failed to encode string to null-terminated C string")]
    StringEncodeError(#[from] NulError),
    #[error("failed to decode string to UTF8")]
    StringDecodeError(#[from] Utf8Error),
    #[error("failed to construct model: {0}")]
    ConstructFailedError(String),
    #[error("failed during evaluation: {0}")]
    EvaluationFailed(String),
}

pub fn open_library(path: &std::path::Path) -> Result<ffi::Bridgestan, libloading::Error> {
    unsafe { ffi::Bridgestan::new(path) }
}

/// Safe wrapper for BridgeStan C functions
#[non_exhaustive]
pub struct StanModel<'lib> {
    model: *mut ffi::bs_model_rng,
    lib: &'lib ffi::Bridgestan,
}

// BridgeStan model is thread safe
unsafe impl<'lib> Sync for StanModel<'lib> {}

impl<'lib> StanModel<'lib> {
    /// Create a new instance of the compiled Stan model.
    /// Data is specified as a JSON file at the given path, or empty for no data
    /// Seed and chain ID are used for reproducibility.
    pub fn new(
        lib: &'lib ffi::Bridgestan,
        path: &str,
        seed: u32,
        chain_id: u32,
    ) -> Result<Self, BridgeStanError> {
        let data = CString::new(path)?.into_raw();

        let mut err: *mut std::os::raw::c_char = std::ptr::null_mut();

        let model = unsafe { lib.bs_construct(data, seed, chain_id, &mut err) };

        // retake pointer to free memory
        let _ = unsafe { CString::from_raw(data) };

        if model.is_null() {
            if err.is_null() {
                Err(BridgeStanError::ConstructFailedError(
                    "Unknown error".to_string(),
                ))
            } else {
                let error_text = String::from(unsafe { CStr::from_ptr(err) }.to_str()?);
                unsafe { lib.bs_free_error_msg(err) };
                Err(BridgeStanError::ConstructFailedError(error_text))
            }
        } else {
            Ok(StanModel { model, lib })
        }
    }

    /// Return the name of the model or error if UTF decode fails
    pub fn name(&self) -> Result<&str, BridgeStanError> {
        let cstr = unsafe { CStr::from_ptr(self.lib.bs_name(self.model)) };
        let res = cstr.to_str()?;
        Ok(res)
    }

    /// Number of parameters in the model on the constrained scale.
    /// Will also count transformed parameters and generated quantities if requested
    pub fn param_num(&self, include_tp: bool, include_gq: bool) -> usize {
        unsafe {
            self.lib
                .bs_param_num(self.model, include_tp as i32, include_gq as i32)
        }
        .try_into()
        .unwrap()
    }

    /// Return the number of parameters on the unconstrained scale.
    /// In particular, this is the size of the slice required by the log_density functions.
    pub fn param_unc_num(&self) -> usize {
        unsafe { self.lib.bs_param_unc_num(self.model) }
            .try_into()
            .unwrap()
    }

    pub fn log_density_gradient<'a>(
        &self,
        theta: &[f64],
        propto: bool,
        jacobian: bool,
        out: &'a mut [f64],
    ) -> Result<(f64, &'a mut [f64]), BridgeStanError> {
        let n = self.param_unc_num();
        assert_eq!(
            theta.len(),
            n,
            "Argument 'theta' must be the same size as the number of parameters!"
        );
        assert_eq!(
            out.len(),
            n,
            "Argument 'out' must be the same size as the number of parameters!"
        );

        let mut val = 0.0;

        let mut err: *mut std::os::raw::c_char = std::ptr::null_mut();
        let rc = unsafe {
            self.lib.bs_log_density_gradient(
                self.model,
                propto as i32,
                jacobian as i32,
                theta.as_ptr(),
                &mut val,
                out.as_mut_ptr(),
                &mut err,
            )
        };

        if rc == 0 {
            Ok((val, out))
        } else {
            Err(self.handle_error(err))
        }
    }

    pub fn param_constrain<'a>(
        &self,
        theta_unc: &[f64],
        include_tp: bool,
        include_gq: bool,
        out: &'a mut [f64],
    ) -> Result<&'a mut [f64], BridgeStanError> {
        let n = self.param_unc_num();
        assert_eq!(
            theta_unc.len(),
            n,
            "Argument 'theta_unc' must be the same size as the number of parameters!"
        );
        let out_n = self.param_num(include_tp, include_gq);
        assert_eq!(
            out.len(),
            out_n,
            "Argument 'out' must be the same size as the number of parameters!"
        );

        let mut err: *mut std::os::raw::c_char = std::ptr::null_mut();
        let rc = unsafe {
            self.lib.bs_param_constrain(
                self.model,
                include_tp as i32,
                include_gq as i32,
                theta_unc.as_ptr(),
                out.as_mut_ptr(),
                &mut err,
            )
        };

        if rc == 0 {
            Ok(out)
        } else {
            Err(self.handle_error(err))
        }
    }

    // etc, need more functions

    fn handle_error(&self, error: *mut std::os::raw::c_char) -> BridgeStanError {
        if error.is_null() {
            return BridgeStanError::EvaluationFailed("Unknown error".to_string());
        }
        let error_text = unsafe { CStr::from_ptr(error) }.to_str().map(String::from);
        unsafe { self.lib.bs_free_error_msg(error) };
        match error_text {
            Ok(s) => BridgeStanError::EvaluationFailed(s),
            Err(_) => BridgeStanError::EvaluationFailed("Unknown error".to_string()),
        }
    }
}

impl<'lib> Drop for StanModel<'lib> {
    /// Free the memory allocated in C++.
    fn drop(&mut self) {
        unsafe { self.lib.bs_destruct(self.model) }
    }
}

#[cfg(feature = "nuts")]
use nuts_rs::{CpuLogpFunc, LogpError};

#[cfg(feature = "nuts")]
impl LogpError for BridgeStanError {
    fn is_recoverable(&self) -> bool {
        !matches!(self, BridgeStanError::EvaluationFailed(_))
    }
}

#[cfg(feature = "nuts")]
impl<'lib> CpuLogpFunc for StanModel<'lib> {
    type Err = BridgeStanError;

    fn dim(&self) -> usize {
        self.param_unc_num()
    }

    fn logp(&mut self, position: &[f64], grad: &mut [f64]) -> Result<f64, Self::Err> {
        let (logp, _grad) = self.log_density_gradient(position, false, true, grad)?;

        Ok(logp)
    }
}
